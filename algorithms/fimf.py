from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


def _entropy(values: np.ndarray) -> float:
    """计算离散变量的香农熵（以比特为单位）。"""
    if values.ndim != 1:
        raise ValueError("熵计算要求输入为一维向量")

    unique, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(np.float64) / values.size
    # 避免对0取对数
    mask = probabilities > 0
    if not np.any(mask):
        return 0.0
    return float(-np.sum(probabilities[mask] * np.log2(probabilities[mask])))


def _joint_entropy(columns: np.ndarray) -> float:
    """计算多列联合熵。"""
    if columns.ndim != 2:
        raise ValueError("联合熵计算要求输入为二维矩阵")
    if columns.shape[0] == 0:
        return 0.0
    unique, counts = np.unique(columns, axis=0, return_counts=True)
    probabilities = counts.astype(np.float64) / columns.shape[0]
    mask = probabilities > 0
    if not np.any(mask):
        return 0.0
    return float(-np.sum(probabilities[mask] * np.log2(probabilities[mask])))


@dataclass
class FIMFFeatureSelector:
    """基于文档算法的多标签特征选择器。

    参数
    ----
    n_selected:
        最终需要保留的特征数量。
    b:
        依赖关系的最大阶数（文中记为 b，需满足 b >= 2）。
    q_ratio:
        选取高熵标签的比例，用于构造集合 Q。最终 |Q| = max(2, round(q_ratio * |L|))。
    n_bins:
        特征离散化时的分箱数量。
    random_state:
        随机种子，仅用于离散化中的排序一致性（KBinsDiscretizer）。
    """

    n_selected: int
    b: int = 2
    q_ratio: float = 0.5
    n_bins: int = 10
    random_state: int | None = None

    feature_scores_: np.ndarray = field(init=False, default=None)
    selected_indices_: np.ndarray = field(init=False, default=None)
    _feature_entropy: np.ndarray = field(init=False, default=None)
    _label_entropy: np.ndarray = field(init=False, default=None)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "FIMFFeatureSelector":
        if self.b < 2:
            raise ValueError("参数 b 必须大于等于 2")
        if not 0 < self.q_ratio <= 1:
            raise ValueError("q_ratio 必须位于 (0, 1] 区间")

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.int32)
        n_samples, n_features = X.shape
        _, n_labels = Y.shape

        if self.n_selected <= 0:
            raise ValueError("n_selected 必须为正整数")
        if self.n_selected > n_features:
            raise ValueError("n_selected 不能超过特征总数")

        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
        )
        X_discrete = discretizer.fit_transform(X).astype(np.int32)

        self._feature_entropy = np.array(
            [_entropy(X_discrete[:, idx]) for idx in range(n_features)],
            dtype=np.float64,
        )

        # 仅保留熵大于0的特征，避免冗余计算
        candidate_indices = [idx for idx, h in enumerate(self._feature_entropy) if h > 0]
        if not candidate_indices:
            # 如果所有特征熵为0，则返回任意n_selected个特征
            self.selected_indices_ = np.arange(self.n_selected, dtype=np.int32)
            self.feature_scores_ = np.zeros(n_features, dtype=np.float64)
            self._label_entropy = np.zeros(n_labels, dtype=np.float64)
            return self

        label_entropy = np.array([_entropy(Y[:, idx]) for idx in range(n_labels)], dtype=np.float64)
        self._label_entropy = label_entropy

        q_size = int(np.ceil(self.q_ratio * n_labels))
        q_size = min(max(q_size, 2), n_labels)
        q_indices = np.argsort(-label_entropy)[:q_size]

        # 预先缓存联合熵
        joint_cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}

        def get_joint_entropy(feature_idx: int, label_subset: Sequence[int]) -> float:
            key = (feature_idx, tuple(sorted(label_subset)))
            if key not in joint_cache:
                columns = np.column_stack(
                    (X_discrete[:, feature_idx], Y[:, list(label_subset)]),
                )
                joint_cache[key] = _joint_entropy(columns)
            return joint_cache[key]

        def tilde_I(feature_idx: int, label_subset: Sequence[int]) -> float:
            base = self._feature_entropy[feature_idx]
            total = base
            subset_size = len(label_subset)
            if subset_size == 0:
                return base
            for r in range(1, subset_size + 1):
                sign = (-1) ** r
                for comb in itertools.combinations(label_subset, r):
                    total += sign * get_joint_entropy(feature_idx, comb)
            return total

        scores = np.zeros(n_features, dtype=np.float64)

        label_indices = list(range(n_labels))

        for feature_idx in candidate_indices:
            tilde_v2 = sum(tilde_I(feature_idx, [label_idx]) for label_idx in label_indices)

            score = tilde_v2
            if self.b >= 2:
                for k in range(3, self.b + 2):
                    subset_size = k - 1
                    if subset_size > len(q_indices):
                        break
                    vk = sum(
                        tilde_I(feature_idx, combo)
                        for combo in itertools.combinations(q_indices, subset_size)
                    )
                    score += ((-1) ** k) * vk
            scores[feature_idx] = score

        # 如果候选特征不足，允许从剩余熵为0的特征中补足
        sorted_indices = np.argsort(-scores)
        selected = []
        for idx in sorted_indices:
            if scores[idx] == 0.0 and idx not in candidate_indices:
                continue
            selected.append(idx)
            if len(selected) == self.n_selected:
                break

        if len(selected) < self.n_selected:
            remaining = [idx for idx in range(n_features) if idx not in selected]
            selected.extend(remaining[: self.n_selected - len(selected)])

        self.selected_indices_ = np.array(selected, dtype=np.int32)
        self.feature_scores_ = scores
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("请先调用 fit 方法")
        return X[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.fit(X, Y).transform(X)
