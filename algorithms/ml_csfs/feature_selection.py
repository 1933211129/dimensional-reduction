"""多标签代价敏感特征选择算法实现。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .data_utils import mean_impute

__all__ = ["PositiveRegionResult", "CostSensitiveFeatureSelector"]


@dataclass(slots=True)
class PositiveRegionResult:
    mask: np.ndarray
    size: int
    delta: float


class CostSensitiveFeatureSelector:
    """基于自适应阈值的多标签代价敏感特征选择算法。"""

    def __init__(
        self,
        alpha: float = 0.5,
        lambda_param: float = 1.5,
        base_neighbors: int = 10,
        max_features: int | None = None,
        n_jobs: int = -1,
        positive_fraction_schedule: Sequence[float] | None = None,
        neighbor_multiplier: float = 3.0,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha 取值范围为 (0, 1]。")
        if lambda_param <= 0:
            raise ValueError("lambda_param 必须为正。")
        if base_neighbors < 1:
            raise ValueError("base_neighbors 必须不小于 1。")
        if positive_fraction_schedule is None:
            positive_fraction_schedule = (1.0, 0.75, 0.5, 0.25, 0.0)
        if len(positive_fraction_schedule) == 0:
            raise ValueError("positive_fraction_schedule 不应为空。")
        validated_schedule: List[float] = []
        for value in positive_fraction_schedule:
            if not 0.0 <= value <= 1.0:
                raise ValueError("positive_fraction_schedule 中的值必须位于 [0, 1]。")
            validated_schedule.append(float(value))
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.base_neighbors = base_neighbors
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.positive_fraction_schedule = tuple(dict.fromkeys(validated_schedule))
        self.neighbor_multiplier = float(neighbor_multiplier) if neighbor_multiplier >= 1.0 else 1.0

        self._cache_enabled: bool = False

    # 拟合结果属性
    selected_indices_: List[int]
    core_indices_: List[int]
    feature_costs_: np.ndarray
    col_means_: np.ndarray
    positive_region_full_: np.ndarray
    positive_region_size_full_: int
    runtime_seconds_: float

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_costs: Sequence[float] | None = None,
    ) -> "CostSensitiveFeatureSelector":
        """执行特征选择。

        参数
        ----
        X : np.ndarray
            特征矩阵，允许包含缺失值。
        y : np.ndarray
            多标签0/1矩阵。
        feature_costs : Sequence[float], optional
            每个特征的测试成本；若缺省则使用列标准差归一化表示。
        """
        start_time = perf_counter()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int8)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X 与 y 的样本数量不一致。")

        num_samples, num_features = X.shape
        if num_features == 0:
            raise ValueError("特征数为0，无法进行约简。")

        if np.any((y != 0) & (y != 1)):
            raise ValueError("y 必须为0/1多标签矩阵。")

        X_imputed, col_means = mean_impute(X)
        self.col_means_ = col_means

        if feature_costs is None:
            std = np.nanstd(X, axis=0)
            std = np.where(np.isnan(std), 0.0, std)
            self.feature_costs_ = std + 1.0  # 保证正值
        else:
            costs = np.asarray(feature_costs, dtype=np.float64)
            if costs.shape[0] != num_features:
                raise ValueError("feature_costs 长度与特征数不匹配。")
            if np.any(costs < 0):
                raise ValueError("feature_costs 应全部非负。")
            # 若存在全0，避免归一化时除0
            self.feature_costs_ = costs + 1e-8

        self._label_indices = [np.flatnonzero(row) for row in y]
        self._X_imputed_cached = X_imputed
        self._y_cached = y
        self._all_features_tuple = tuple(range(num_features))

        self._prepare_neighbor_cache(X_imputed, y)

        full_subset = self._all_features_tuple
        full_sqdist = self._full_sqdist if self._cache_enabled else None

        positive_fraction = self.positive_fraction_schedule[0]
        full_region: PositiveRegionResult | None = None
        for candidate_fraction in self.positive_fraction_schedule:
            region_candidate = self._compute_positive_region(
                full_sqdist, full_subset, candidate_fraction
            )
            full_region = region_candidate
            positive_fraction = candidate_fraction
            if region_candidate.size > 0:
                break
        if full_region is None:
            raise RuntimeError("无法计算完整特征集的正域。")
        self.positive_fraction_ = positive_fraction
        self.positive_region_full_ = full_region.mask
        self.positive_region_size_full_ = full_region.size

        core_features: List[int] = []
        all_indices = list(range(num_features))
        for idx in all_indices:
            subset = tuple(j for j in all_indices if j != idx)
            if self._cache_enabled and full_sqdist is not None:
                subset_sqdist = full_sqdist - self._feature_neighbor_sqdist[idx]
            else:
                subset_sqdist = None
            region = self._compute_positive_region(
                subset_sqdist, subset, self.positive_fraction_
            )
            if not np.array_equal(region.mask, self.positive_region_full_):
                core_features.append(idx)

        selected: List[int] = sorted(core_features)
        remaining = [idx for idx in all_indices if idx not in selected]

        if selected:
            if self._cache_enabled:
                current_sqdist = np.add.reduce(
                    self._feature_neighbor_sqdist[selected],
                    axis=0,
                    dtype=np.float32,
                )
            else:
                current_sqdist = None
            current_region = self._compute_positive_region(
                current_sqdist, tuple(selected), self.positive_fraction_
            )
        else:
            current_sqdist = None
            current_region = PositiveRegionResult(
                mask=np.zeros_like(self.positive_region_full_, dtype=bool),
                size=0,
                delta=0.0,
            )

        while not np.array_equal(current_region.mask, self.positive_region_full_):
            if not remaining:
                break

            candidate_indices: List[int] = []
            candidate_regions: List[PositiveRegionResult] = []
            pos_values: List[int] = []
            cost_values: List[float] = []

            for idx in remaining:
                subset = tuple(selected + [idx])
                if self._cache_enabled:
                    if current_sqdist is None:
                        sqdist_candidate = self._feature_neighbor_sqdist[idx]
                    else:
                        sqdist_candidate = (
                            current_sqdist + self._feature_neighbor_sqdist[idx]
                        )
                else:
                    sqdist_candidate = None
                region = self._compute_positive_region(
                    sqdist_candidate, subset, self.positive_fraction_
                )
                candidate_indices.append(idx)
                candidate_regions.append(region)
                pos_values.append(region.size)
                cost_values.append(self.feature_costs_[list(subset)].sum())

            pos_array = np.asarray(pos_values, dtype=float)
            cost_array = np.asarray(cost_values, dtype=float)

            pos_min = pos_array.min()
            pos_max = pos_array.max()
            cost_min = cost_array.min()
            cost_max = cost_array.max()

            best_score = -np.inf
            best_idx = None
            best_region = None

            for idx, pos_size, cost_subset, region in zip(
                candidate_indices, pos_array, cost_array, candidate_regions
            ):
                if pos_max == pos_min:
                    pos_norm = 1.0
                else:
                    pos_norm = (pos_size - pos_min) / (pos_max - pos_min)

                if cost_max == cost_min:
                    cost_norm = 1.0
                else:
                    cost_norm = (cost_subset - cost_min) / (cost_max - cost_min)

                significance = self.alpha * pos_norm - (1.0 - self.alpha) * cost_norm
                if significance > best_score:
                    best_score = significance
                    best_idx = idx
                    best_region = region

            if best_idx is None or best_region is None:
                break

            selected.append(best_idx)
            remaining.remove(best_idx)
            current_region = best_region

            if self._cache_enabled:
                if current_sqdist is None:
                    current_sqdist = self._feature_neighbor_sqdist[best_idx].copy()
                else:
                    current_sqdist = (
                        current_sqdist + self._feature_neighbor_sqdist[best_idx]
                    )

            if self.max_features is not None and len(selected) >= self.max_features:
                break

        self.selected_indices_ = selected
        self.core_indices_ = core_features
        self.runtime_seconds_ = perf_counter() - start_time
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "selected_indices_"):
            raise RuntimeError("请先调用 fit 方法。")
        return np.asarray(X)[:, self.selected_indices_]

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, feature_costs: Sequence[float] | None = None
    ) -> np.ndarray:
        self.fit(X, y, feature_costs=feature_costs)
        return self.transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray | List[int]:
        if not hasattr(self, "selected_indices_"):
            raise RuntimeError("请先调用 fit 方法。")
        if indices:
            return self.selected_indices_
        mask = np.zeros(self.feature_costs_.shape[0], dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    # 私有方法 ---------------------------------------------------------
    def _prepare_neighbor_cache(self, X: np.ndarray, y: np.ndarray) -> None:
        num_samples, num_features = X.shape
        if num_samples <= 1:
            self._cache_enabled = False
            self._neighbor_candidate_indices = None
            self._neighbor_candidate_labels = None
            self._feature_neighbor_sqdist = None
            self._full_sqdist = None
            return

        effective_neighbors = min(self.base_neighbors, num_samples - 1)
        effective_neighbors = max(effective_neighbors, 1)
        candidate_neighbors = int(
            round((effective_neighbors + 1) * self.neighbor_multiplier)
        )
        if candidate_neighbors < effective_neighbors + 1:
            candidate_neighbors = effective_neighbors + 1
        candidate_neighbors = min(num_samples, candidate_neighbors)

        nbrs = NearestNeighbors(
            n_neighbors=candidate_neighbors,
            metric="euclidean",
            algorithm="auto",
            n_jobs=self.n_jobs,
        )
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)

        self._neighbor_candidate_indices = indices.astype(np.int32, copy=False)
        self._neighbor_candidate_labels = y[indices].astype(np.float32, copy=False)

        neighbor_count = self._neighbor_candidate_indices.shape[1]
        feature_sqdist = np.empty(
            (num_features, num_samples, neighbor_count), dtype=np.float32
        )
        for feature_idx in range(num_features):
            values = X[:, feature_idx]
            diff = values[:, None] - values[indices]
            np.square(diff, out=feature_sqdist[feature_idx])

        self._feature_neighbor_sqdist = feature_sqdist
        self._full_sqdist = np.add.reduce(feature_sqdist, axis=0, dtype=np.float32)
        self._cache_enabled = True

    def _compute_positive_region(
        self,
        sqdist: np.ndarray | None,
        subset: Tuple[int, ...],
        min_fraction: float,
    ) -> PositiveRegionResult:
        subset = tuple(subset)
        if not subset:
            empty_mask = np.zeros(self._X_imputed_cached.shape[0], dtype=bool)
            return PositiveRegionResult(mask=empty_mask, size=0, delta=0.0)

        if not self._cache_enabled or sqdist is None:
            return self._compute_positive_region_exact(
                self._X_imputed_cached, self._y_cached, subset, min_fraction
            )

        num_samples = sqdist.shape[0]
        if num_samples <= 1:
            mask = np.ones(num_samples, dtype=bool)
            return PositiveRegionResult(mask=mask, size=num_samples, delta=0.0)

        order = np.argsort(sqdist, axis=1)
        sorted_sqdist = np.take_along_axis(sqdist, order, axis=1)
        kth_index = min(self.base_neighbors, sorted_sqdist.shape[1] - 1)
        kth_dist = np.sqrt(sorted_sqdist[:, kth_index])
        delta = self.lambda_param * float(np.mean(kth_dist))
        if delta <= 0:
            delta = float(np.finfo(np.float32).eps)
        radius_sq = delta * delta

        if np.any(radius_sq > sorted_sqdist[:, -1]):
            return self._compute_positive_region_exact(
                self._X_imputed_cached, self._y_cached, subset, min_fraction
            )

        neighbor_labels_sorted = np.take_along_axis(
            self._neighbor_candidate_labels, order[:, :, None], axis=1
        )

        within = sorted_sqdist <= radius_sq
        counts = np.count_nonzero(within, axis=1)
        positive_mask = counts > 0

        if min_fraction > 0.0:
            neighbor_counts_safe = np.maximum(counts, 1)[:, None]
            within_float = within.astype(np.float32, copy=False)
            label_sums = np.einsum("ij,ijl->il", within_float, neighbor_labels_sorted)
            label_fractions = label_sums / neighbor_counts_safe

            for sample_idx in range(num_samples):
                if not positive_mask[sample_idx]:
                    continue
                labels_pos = self._label_indices[sample_idx]
                if labels_pos.size == 0:
                    continue
                if not np.all(label_fractions[sample_idx, labels_pos] >= min_fraction):
                    positive_mask[sample_idx] = False

        return PositiveRegionResult(
            mask=positive_mask,
            size=int(np.count_nonzero(positive_mask)),
            delta=delta,
        )

    def _compute_positive_region_exact(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subset: Tuple[int, ...],
        min_fraction: float,
    ) -> PositiveRegionResult:
        if not subset:
            empty_mask = np.zeros(X.shape[0], dtype=bool)
            return PositiveRegionResult(mask=empty_mask, size=0, delta=0.0)

        X_sub = X[:, subset]
        num_samples = X_sub.shape[0]

        if num_samples <= 1:
            mask = np.ones(num_samples, dtype=bool)
            return PositiveRegionResult(mask=mask, size=num_samples, delta=0.0)

        n_neighbors = min(self.base_neighbors, num_samples - 1)
        n_neighbors = max(n_neighbors, 1)

        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors + 1,
            metric="euclidean",
            algorithm="auto",
            n_jobs=self.n_jobs,
        )
        nbrs.fit(X_sub)
        distances, _ = nbrs.kneighbors(X_sub)
        kth = distances[:, -1]
        delta = self.lambda_param * float(np.mean(kth))
        if delta <= 0:
            delta = float(np.finfo(np.float32).eps)

        neighbor_indices = nbrs.radius_neighbors(
            X_sub, radius=delta, return_distance=False
        )

        positive_mask = np.ones(num_samples, dtype=bool)
        for i, neighbors in enumerate(neighbor_indices):
            labels_pos = self._label_indices[i]
            if labels_pos.size == 0:
                continue
            if neighbors.size == 0:
                positive_mask[i] = False
                continue
            neighbor_labels = y[neighbors][:, labels_pos]
            if min_fraction <= 0.0:
                continue
            label_fractions = np.mean(neighbor_labels == 1, axis=0)
            if label_fractions.ndim == 0:
                meets_requirement = float(label_fractions) >= min_fraction
            else:
                meets_requirement = bool(np.all(label_fractions >= min_fraction))
            if not meets_requirement:
                positive_mask[i] = False

        return PositiveRegionResult(
            mask=positive_mask,
            size=int(np.count_nonzero(positive_mask)),
            delta=delta,
        )

