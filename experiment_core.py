from __future__ import annotations

import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from algorithm1 import (
    MISSING_VALUE,
    IncompleteMultiLabelDecisionTable,
    algorithm3_multi_label_reducts,
)
from algorithm_loader import (
    load_fimf_module,
    load_mfs_mcdm_module,
    load_mlcsfs_feature_selection_module,
    load_mlknn_module,
)
from algorithms.mcb_ar import build_incomplete_single_label_table, compute_mcb_ar_reducts
from arff_parser import read_arff
from evaluation_metrics import coverage, f1_score_macro, hamming_loss, ranking_loss


DATA_DIR = Path(__file__).resolve().parent / "data"
MISSING_RATIO = 0.05
RANDOM_SEED = 42
DEFAULT_FOLDS = 10
DEFAULT_MISSING_RATIOS = (MISSING_RATIO, 0.10, 0.15)
DEFAULT_K_GRID = (0.01, 0.02, 0.05, 0.10, 0.20)
DEFAULT_CLASSIFIER_IMPUTER = "mean"
MLCSFS_PREFILTER_MAX_FEATURES = 128
MCAR = "mcar"
BY_OBJECT = "by_object"
BY_ATTRIBUTE = "by_attribute"
BLOCKWISE = "blockwise"
MISSINGNESS_PATTERNS = (MCAR, BY_OBJECT, BY_ATTRIBUTE, BLOCKWISE)

OURS = "ours"
MCB_AR = "mcb-ar"
MFS_MCDM = "mfs-mcdm"
FIMF = "fimf"
ML_CSFS = "ml-csfs"
BASELINE_ALGORITHMS = (MCB_AR, MFS_MCDM, FIMF, ML_CSFS)
IMPUTER_STRATEGIES = ("knn", "em", "missforest")


MLkNN = load_mlknn_module().MLkNN
TopsisResult = load_mfs_mcdm_module().TopsisResult
topsis_feature_ranking = load_mfs_mcdm_module().topsis_feature_ranking
FIMFFeatureSelector = load_fimf_module().FIMFFeatureSelector
CostSensitiveFeatureSelector = (
    load_mlcsfs_feature_selection_module().CostSensitiveFeatureSelector
)


@dataclass
class FoldResult:
    fold_index: int
    algorithm: str
    imputer: str
    tuned_k: Optional[int]
    train_objects: int
    validation_objects: int
    test_objects: int
    reduction_time: float
    tuning_time: float
    preprocessing_time: float
    selection_pipeline_time: float
    selected_attributes: int
    reduction_rate: float
    test_f1_score: float
    test_ranking_loss: float
    test_coverage: float
    test_hamming_loss: float
    chosen_features: Tuple[str, ...]


@dataclass
class ExperimentSummary:
    dataset_path: Path
    algorithm: str
    imputer: str
    missing_ratio: float
    missing_pattern: str
    fold_results: List[FoldResult]
    avg_reduction_time: float
    std_reduction_time: float
    avg_tuning_time: float
    std_tuning_time: float
    avg_preprocessing_time: float
    std_preprocessing_time: float
    avg_selection_pipeline_time: float
    std_selection_pipeline_time: float
    avg_selected_attributes: float
    std_selected_attributes: float
    avg_reduction_rate: float
    std_reduction_rate: float
    avg_f1_score: float
    std_f1_score: float
    avg_ranking_loss: float
    std_ranking_loss: float
    avg_coverage: float
    std_coverage: float
    avg_hamming_loss: float
    std_hamming_loss: float
    total_runtime: float


@dataclass
class ComparisonResult:
    dataset_name: str
    missing_ratio: float
    missing_pattern: str
    summaries: Dict[str, ExperimentSummary]


@dataclass
class SignificanceRecord:
    metric: str
    baseline: str
    statistic: float
    p_value: float
    ours_mean: float
    baseline_mean: float


@dataclass
class SelectionResult:
    selected_indices: List[int]
    selected_names: Tuple[str, ...]
    tuned_k: Optional[int]
    imputer: str
    reduction_time: float
    tuning_time: float
    preprocessing_time: float
    selection_pipeline_time: float


@dataclass
class BaselineTuningConfig:
    imputer: str
    tuned_k: int


def configure_parallel_runtime(
    max_workers: int,
    parallel_threshold: int,
    chunk_size: Optional[int],
) -> None:
    os.environ["ALGORITHM1_PARALLEL_THRESHOLD"] = str(max(1, parallel_threshold))
    os.environ["ALGORITHM1_MAX_WORKERS"] = str(max(1, max_workers))
    if chunk_size is not None:
        os.environ["ALGORITHM1_MAP_CHUNK_SIZE"] = str(max(1, chunk_size))
    else:
        os.environ.pop("ALGORITHM1_MAP_CHUNK_SIZE", None)


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    values_list = list(values)
    if not values_list:
        return 0.0, 0.0
    if len(values_list) == 1:
        return float(values_list[0]), 0.0
    return float(statistics.mean(values_list)), float(statistics.stdev(values_list))


def format_numeric(value: float) -> str:
    if value != value:
        return MISSING_VALUE
    formatted = f"{value:.10g}"
    return formatted if formatted else "0"


def matrix_to_incomplete_rows(matrix: np.ndarray) -> List[List[str]]:
    return [
        [MISSING_VALUE if np.isnan(value) else format_numeric(float(value)) for value in row]
        for row in matrix
    ]


def inject_missing_values(
    features: Sequence[Sequence[float]] | np.ndarray,
    missing_ratio: float,
    seed: int,
    *,
    pattern: str = MCAR,
) -> Tuple[np.ndarray, int]:
    if not 0 <= missing_ratio < 1:
        raise ValueError("missing_ratio 必须位于 [0, 1) 区间。")
    if pattern not in MISSINGNESS_PATTERNS:
        raise ValueError(f"不支持的缺失模式：{pattern}")

    feature_matrix = np.asarray(features, dtype=np.float64)
    rng = np.random.default_rng(seed)
    result = feature_matrix.copy()
    target_missing = int(round(feature_matrix.size * missing_ratio))
    if target_missing == 0:
        return result, 0

    object_count, feature_count = feature_matrix.shape

    if pattern == MCAR:
        flat_indices = rng.choice(feature_matrix.size, size=target_missing, replace=False)
    elif pattern == BY_OBJECT:
        row_count = min(object_count, max(1, math.ceil(target_missing / max(1, feature_count))))
        row_indices = np.sort(rng.choice(object_count, size=row_count, replace=False))
        candidate = np.array(
            [row_idx * feature_count + col_idx for row_idx in row_indices for col_idx in range(feature_count)],
            dtype=np.int64,
        )
        flat_indices = rng.choice(candidate, size=target_missing, replace=False)
    elif pattern == BY_ATTRIBUTE:
        col_count = min(feature_count, max(1, math.ceil(target_missing / max(1, object_count))))
        col_indices = np.sort(rng.choice(feature_count, size=col_count, replace=False))
        candidate = np.array(
            [row_idx * feature_count + col_idx for row_idx in range(object_count) for col_idx in col_indices],
            dtype=np.int64,
        )
        flat_indices = rng.choice(candidate, size=target_missing, replace=False)
    else:
        row_count = max(1, min(object_count, int(round(math.sqrt(target_missing * object_count / max(1, feature_count))))))
        col_count = max(1, min(feature_count, int(math.ceil(target_missing / row_count))))
        while row_count * col_count < target_missing:
            if row_count < object_count:
                row_count += 1
            elif col_count < feature_count:
                col_count += 1
            else:
                break
        row_indices = np.sort(rng.choice(object_count, size=row_count, replace=False))
        col_indices = np.sort(rng.choice(feature_count, size=col_count, replace=False))
        candidate = np.array(
            [row_idx * feature_count + col_idx for row_idx in row_indices for col_idx in col_indices],
            dtype=np.int64,
        )
        flat_indices = rng.choice(candidate, size=target_missing, replace=False)

    row_indices, col_indices = np.unravel_index(flat_indices, result.shape)
    result[row_indices, col_indices] = np.nan
    injected_count = int(target_missing)
    return result, injected_count


def build_kfold_indices(
    object_count: int,
    folds: int,
    seed: int,
    overlap_ratio: float = 0.0,
) -> List[Tuple[List[int], List[int]]]:
    if folds < 1:
        raise ValueError("折数必须不少于 1。")
    if folds > object_count:
        raise ValueError("折数不能超过样本数量。")
    if overlap_ratio not in (0.0, 0):
        raise ValueError("当前实验协议固定为标准不重叠 K-fold，overlap_ratio 必须为 0。")

    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    indices = np.arange(object_count)
    return [
        (train_idx.tolist(), test_idx.tolist())
        for train_idx, test_idx in splitter.split(indices)
    ]


def _train_validation_split_indices(
    train_indices: Sequence[int],
    seed: int,
    validation_ratio: float = 0.2,
) -> Tuple[List[int], List[int]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio 必须位于 (0, 1) 区间。")

    indices = list(train_indices)
    rng = random.Random(seed)
    rng.shuffle(indices)

    validation_size = max(1, int(round(len(indices) * validation_ratio)))
    validation_size = min(validation_size, len(indices) - 1)
    validation_indices = sorted(indices[:validation_size])
    inner_train_indices = sorted(indices[validation_size:])
    return inner_train_indices, validation_indices


def _sample_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    sample_fraction: float,
    max_objects: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    object_count = features.shape[0]
    if not (0 < sample_fraction <= 1.0):
        raise ValueError("sample_fraction 必须位于 (0, 1] 区间。")

    target_objects = object_count
    if sample_fraction < 1.0:
        target_objects = min(
            target_objects,
            max(1, int(math.ceil(object_count * sample_fraction))),
        )
    if max_objects is not None:
        if max_objects <= 0:
            raise ValueError("max_objects 必须为正整数。")
        target_objects = min(target_objects, max_objects)

    if target_objects == object_count:
        return features, labels

    rng = random.Random(seed)
    indices = list(range(object_count))
    rng.shuffle(indices)
    selected_indices = sorted(indices[:target_objects])
    return features[selected_indices], labels[selected_indices]


def _label_rows_to_sets(
    label_matrix: np.ndarray,
    label_names: Sequence[str],
) -> List[set[str]]:
    return [
        {label_names[label_idx] for label_idx, value in enumerate(row) if value == 1}
        for row in np.asarray(label_matrix, dtype=np.int8)
    ]


def _probabilities_to_score_dicts(
    probabilities: np.ndarray,
    label_names: Sequence[str],
) -> List[Dict[str, float]]:
    return [
        {label_names[label_idx]: float(score) for label_idx, score in enumerate(row)}
        for row in np.asarray(probabilities, dtype=np.float64)
    ]


def _predictions_to_sets(
    predictions: np.ndarray,
    label_names: Sequence[str],
) -> List[set[str]]:
    return [
        {label_names[label_idx] for label_idx, value in enumerate(row) if value == 1}
        for row in np.asarray(predictions, dtype=np.int8)
    ]


def _classifier_impute(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    strategy: str = DEFAULT_CLASSIFIER_IMPUTER,
) -> Tuple[np.ndarray, np.ndarray]:
    if X_train.shape[1] == 0:
        return (
            np.zeros((X_train.shape[0], 1), dtype=np.float32),
            np.zeros((X_test.shape[0], 1), dtype=np.float32),
        )
    if strategy != "mean":
        raise ValueError("当前分类器阶段仅支持均值插补。")
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed.astype(np.float32), X_test_imputed.astype(np.float32)


def _prepare_classifier_data(
    *,
    algorithm: str,
    imputer_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if algorithm in (OURS, MCB_AR):
        return _classifier_impute(
            X_train,
            X_test,
            strategy=DEFAULT_CLASSIFIER_IMPUTER,
        )

    train_imputed, [test_imputed] = _fit_transform_imputer(
        imputer_name,
        X_train,
        [X_test],
        seed,
    )
    return train_imputed, test_imputed


def _build_imputer(strategy: str, seed: int):
    if strategy == "knn":
        return KNNImputer(n_neighbors=3, weights="distance")
    if strategy == "em":
        return IterativeImputer(
            estimator=BayesianRidge(),
            random_state=seed,
            sample_posterior=False,
            max_iter=4,
            n_nearest_features=16,
            skip_complete=True,
            initial_strategy="mean",
        )
    if strategy == "missforest":
        return IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=10,
                random_state=seed,
                n_jobs=-1,
            ),
            random_state=seed,
            sample_posterior=False,
            max_iter=3,
            n_nearest_features=12,
            skip_complete=True,
            initial_strategy="mean",
        )
    raise ValueError(f"不支持的插补策略：{strategy}")


def _fit_transform_imputer(
    strategy: str,
    X_train: np.ndarray,
    X_other: Sequence[np.ndarray],
    seed: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    imputer = _build_imputer(strategy, seed)
    train_imputed = imputer.fit_transform(X_train)
    transformed = [imputer.transform(matrix) for matrix in X_other]
    return train_imputed.astype(np.float32), [matrix.astype(np.float32) for matrix in transformed]


def _candidate_k_values(feature_count: int) -> List[int]:
    values = {
        max(1, min(feature_count, int(math.ceil(feature_count * ratio))))
        for ratio in DEFAULT_K_GRID
    }
    return sorted(values)


def _evaluate_with_mlknn(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    label_names: Sequence[str],
    *,
    k: int = 10,
) -> Dict[str, float]:
    classifier = MLkNN(k=k)
    classifier.fit(train_X, train_y)
    result = classifier.predict_with_prob(test_X)

    predictions = _predictions_to_sets(result.predictions, label_names)
    truths = _label_rows_to_sets(test_y, label_names)
    score_dicts = _probabilities_to_score_dicts(result.probabilities, label_names)

    return {
        "f1_score": f1_score_macro(predictions, truths),
        "ranking_loss": ranking_loss(score_dicts, truths),
        "coverage": coverage(score_dicts, truths),
        "hamming_loss": hamming_loss(predictions, truths),
    }


def _select_with_ours(
    *,
    train_X_missing: np.ndarray,
    train_y: np.ndarray,
    feature_names: Sequence[str],
    label_names: Sequence[str],
    max_reducts: Optional[int],
    timeout: Optional[float],
    prefer_greedy: bool,
) -> SelectionResult:
    start = time.perf_counter()
    table = IncompleteMultiLabelDecisionTable(
        condition_attributes=tuple(feature_names),
        condition_values=matrix_to_incomplete_rows(train_X_missing),
        label_names=tuple(label_names),
        label_values=train_y.astype(np.int8).tolist(),
    )
    reducts = algorithm3_multi_label_reducts(
        table,
        max_reducts=max_reducts,
        timeout=timeout,
        prefer_greedy=prefer_greedy,
    )
    if not reducts:
        raise RuntimeError("Ours 未返回任何约简。")

    chosen = sorted(reducts, key=lambda subset: (len(subset), sorted(subset)))[0]
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    selected_indices = sorted(name_to_index[name] for name in chosen)
    return SelectionResult(
        selected_indices=selected_indices,
        selected_names=tuple(feature_names[idx] for idx in selected_indices),
        tuned_k=len(selected_indices),
        imputer="native",
        reduction_time=time.perf_counter() - start,
        tuning_time=0.0,
        preprocessing_time=0.0,
        selection_pipeline_time=time.perf_counter() - start,
    )


def _label_rows_to_single_decisions(label_matrix: np.ndarray) -> List[str]:
    decisions: List[str] = []
    for row in np.asarray(label_matrix, dtype=np.int8):
        decisions.append("".join(str(int(value)) for value in row))
    return decisions


def _select_with_mcb_ar(
    *,
    train_X_missing: np.ndarray,
    train_y: np.ndarray,
    feature_names: Sequence[str],
) -> SelectionResult:
    start = time.perf_counter()
    table = build_incomplete_single_label_table(
        condition_attributes=tuple(feature_names),
        condition_values=matrix_to_incomplete_rows(train_X_missing),
        decision_values=_label_rows_to_single_decisions(train_y),
        decision_attribute="label_signature",
    )
    reducts = compute_mcb_ar_reducts(table, max_reducts=1)
    if not reducts:
        raise RuntimeError("MCB-AR 未返回任何约简。")

    chosen = sorted(reducts, key=lambda subset: (len(subset), sorted(subset)))[0]
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    selected_indices = sorted(name_to_index[name] for name in chosen)
    elapsed = time.perf_counter() - start
    return SelectionResult(
        selected_indices=selected_indices,
        selected_names=tuple(feature_names[idx] for idx in selected_indices),
        tuned_k=len(selected_indices),
        imputer="native",
        reduction_time=elapsed,
        tuning_time=0.0,
        preprocessing_time=0.0,
        selection_pipeline_time=elapsed,
    )


def _select_mfs_topk(
    feature_names: Sequence[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> List[int]:
    ranking: TopsisResult = topsis_feature_ranking(
        feature_names=feature_names,
        feature_rows=X_train.tolist(),
        label_rows=y_train.astype(np.int8).tolist(),
        missing_token=MISSING_VALUE,
        lambda_reg=10.0,
        impute_strategy="mean",
    )
    return sorted(ranking.ranking_indices[:k])


def _select_fimf_topk(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    seed: int,
) -> List[int]:
    selector = FIMFFeatureSelector(
        n_selected=k,
        random_state=seed,
    )
    selector.fit(X_train, y_train)
    return sorted(int(idx) for idx in selector.selected_indices_)


def _rank_fimf_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_k: int,
    seed: int,
) -> List[int]:
    selector = FIMFFeatureSelector(
        n_selected=max_k,
        random_state=seed,
    )
    selector.fit(X_train, y_train)
    return list(np.argsort(-selector.feature_scores_))


def _select_mlcsfs_topk(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> List[int]:
    return sorted(_rank_mlcsfs_features(X_train, y_train, k)[:k])


def _variance_prefilter_indices(
    X_train: np.ndarray,
    *,
    max_features: int,
) -> List[int]:
    feature_count = X_train.shape[1]
    if feature_count <= max_features:
        return list(range(feature_count))

    variances = np.var(X_train, axis=0, dtype=np.float64)
    ranking = np.argsort(-variances, kind="mergesort")
    selected = sorted(int(idx) for idx in ranking[:max_features])
    return selected


def _rank_mlcsfs_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_k: int,
) -> List[int]:
    prefilter_indices = _variance_prefilter_indices(
        X_train,
        max_features=max(max_k, MLCSFS_PREFILTER_MAX_FEATURES),
    )
    X_prefiltered = X_train[:, prefilter_indices]
    selector = CostSensitiveFeatureSelector(
        lambda_param=1.5,
        max_features=max_k,
    )
    selector.fit(X_prefiltered, y_train)
    return [prefilter_indices[int(idx)] for idx in selector.selected_indices_]


def _baseline_selector_indices(
    algorithm: str,
    feature_names: Sequence[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    seed: int,
) -> List[int]:
    if algorithm == MCB_AR:
        raise ValueError("MCB-AR 不支持基于 top-k 的 baseline selector 接口。")
    if algorithm == MFS_MCDM:
        return _select_mfs_topk(feature_names, X_train, y_train, k)
    if algorithm == FIMF:
        return _select_fimf_topk(X_train, y_train, k, seed)
    if algorithm == ML_CSFS:
        return _select_mlcsfs_topk(X_train, y_train, k)
    raise ValueError(f"未知 baseline 算法：{algorithm}")


def _tune_baseline_config(
    *,
    algorithm: str,
    train_X_missing: np.ndarray,
    train_y: np.ndarray,
    feature_names: Sequence[str],
    label_names: Sequence[str],
    inner_seed: int,
) -> Tuple[BaselineTuningConfig, float]:
    inner_train_indices, validation_indices = _train_validation_split_indices(
        list(range(train_X_missing.shape[0])),
        inner_seed,
    )
    X_inner_train = train_X_missing[inner_train_indices]
    y_inner_train = train_y[inner_train_indices]
    X_validation = train_X_missing[validation_indices]
    y_validation = train_y[validation_indices]

    candidate_ks = _candidate_k_values(len(feature_names))
    best_config: Optional[BaselineTuningConfig] = None
    best_key: Optional[Tuple[float, float, int, str]] = None

    tuning_start = time.perf_counter()
    for imputer_name in IMPUTER_STRATEGIES:
        inner_imputed, [validation_imputed] = _fit_transform_imputer(
            imputer_name,
            X_inner_train,
            [X_validation],
            inner_seed,
        )

        mfs_ranking: Optional[List[int]] = None
        fimf_ranking: Optional[List[int]] = None
        mlcsfs_ranking: Optional[List[int]] = None
        if algorithm == MFS_MCDM:
            ranking: TopsisResult = topsis_feature_ranking(
                feature_names=feature_names,
                feature_rows=inner_imputed.tolist(),
                label_rows=y_inner_train.astype(np.int8).tolist(),
                missing_token=MISSING_VALUE,
                lambda_reg=10.0,
                impute_strategy="mean",
            )
            mfs_ranking = list(ranking.ranking_indices)
        elif algorithm == FIMF:
            fimf_ranking = _rank_fimf_features(
                inner_imputed,
                y_inner_train,
                max(candidate_ks),
                inner_seed,
            )
        elif algorithm == ML_CSFS:
            mlcsfs_ranking = _rank_mlcsfs_features(
                inner_imputed,
                y_inner_train,
                max(candidate_ks),
            )

        evaluated_subsets: Dict[Tuple[int, ...], Dict[str, float]] = {}
        for k in candidate_ks:
            if algorithm == MFS_MCDM and mfs_ranking is not None:
                selected_indices = sorted(int(idx) for idx in mfs_ranking[:k])
            elif algorithm == FIMF and fimf_ranking is not None:
                selected_indices = sorted(int(idx) for idx in fimf_ranking[:k])
            elif algorithm == ML_CSFS and mlcsfs_ranking is not None:
                selected_indices = sorted(int(idx) for idx in mlcsfs_ranking[:k])
            else:
                selected_indices = _baseline_selector_indices(
                    algorithm,
                    feature_names,
                    inner_imputed,
                    y_inner_train,
                    k,
                    inner_seed,
                )
            subset_key = tuple(selected_indices)
            metrics = evaluated_subsets.get(subset_key)
            if metrics is None:
                selected_validation = validation_imputed[:, selected_indices]
                selected_train = inner_imputed[:, selected_indices]
                metrics = _evaluate_with_mlknn(
                    selected_train,
                    y_inner_train,
                    selected_validation,
                    y_validation,
                    label_names,
                )
                evaluated_subsets[subset_key] = metrics
            current_key = (
                float(metrics["ranking_loss"]),
                -float(metrics["f1_score"]),
                len(selected_indices),
                imputer_name,
            )
            if best_key is None or current_key < best_key:
                best_key = current_key
                best_config = BaselineTuningConfig(
                    imputer=imputer_name,
                    tuned_k=len(selected_indices),
                )

    if best_config is None:
        raise RuntimeError(f"baseline {algorithm} 在验证阶段未产生有效选择结果。")

    return best_config, time.perf_counter() - tuning_start


def _refit_baseline_selection(
    *,
    algorithm: str,
    train_X_missing: np.ndarray,
    train_y: np.ndarray,
    feature_names: Sequence[str],
    tuning_config: BaselineTuningConfig,
    seed: int,
    tuning_time: float,
) -> SelectionResult:
    preprocessing_start = time.perf_counter()
    outer_train_imputed, _ = _fit_transform_imputer(
        tuning_config.imputer,
        train_X_missing,
        [],
        seed,
    )
    preprocessing_time = time.perf_counter() - preprocessing_start

    selection_start = time.perf_counter()
    selected_indices = _baseline_selector_indices(
        algorithm,
        feature_names,
        outer_train_imputed,
        train_y,
        tuning_config.tuned_k,
        seed,
    )
    reduction_time = time.perf_counter() - selection_start
    selected_names = tuple(feature_names[idx] for idx in selected_indices)

    return SelectionResult(
        selected_indices=selected_indices,
        selected_names=selected_names,
        tuned_k=len(selected_indices),
        imputer=tuning_config.imputer,
        reduction_time=reduction_time,
        tuning_time=tuning_time,
        preprocessing_time=preprocessing_time,
        selection_pipeline_time=tuning_time + preprocessing_time + reduction_time,
    )


def _tune_baseline(
    *,
    algorithm: str,
    train_X_missing: np.ndarray,
    train_y: np.ndarray,
    feature_names: Sequence[str],
    label_names: Sequence[str],
    inner_seed: int,
) -> SelectionResult:
    tuning_config, tuning_time = _tune_baseline_config(
        algorithm=algorithm,
        train_X_missing=train_X_missing,
        train_y=train_y,
        feature_names=feature_names,
        label_names=label_names,
        inner_seed=inner_seed,
    )
    return _refit_baseline_selection(
        algorithm=algorithm,
        train_X_missing=train_X_missing,
        train_y=train_y,
        feature_names=feature_names,
        tuning_config=tuning_config,
        seed=inner_seed,
        tuning_time=tuning_time,
    )


def _select_features_for_algorithm(
    *,
    algorithm: str,
    train_X_missing: np.ndarray,
    train_y: np.ndarray,
    feature_names: Sequence[str],
    label_names: Sequence[str],
    seed: int,
    max_reducts: Optional[int],
    timeout: Optional[float],
    prefer_greedy: bool,
) -> SelectionResult:
    if algorithm == OURS:
        return _select_with_ours(
            train_X_missing=train_X_missing,
            train_y=train_y,
            feature_names=feature_names,
            label_names=label_names,
            max_reducts=max_reducts,
            timeout=timeout,
            prefer_greedy=prefer_greedy,
        )
    if algorithm == MCB_AR:
        return _select_with_mcb_ar(
            train_X_missing=train_X_missing,
            train_y=train_y,
            feature_names=feature_names,
        )

    return _tune_baseline(
        algorithm=algorithm,
        train_X_missing=train_X_missing,
        train_y=train_y,
        feature_names=feature_names,
        label_names=label_names,
        inner_seed=seed,
    )


def run_algorithm_experiment(
    *,
    data_path: Path,
    algorithm: str = OURS,
    folds: int = DEFAULT_FOLDS,
    missing_ratio: float = MISSING_RATIO,
    seed: int = RANDOM_SEED,
    max_workers: Optional[int] = None,
    parallel_threshold: int = 1,
    chunk_size: Optional[int] = None,
    title: Optional[str] = None,
    sample_fraction: float = 1.0,
    max_objects: Optional[int] = None,
    sample_seed: Optional[int] = None,
    max_reducts: Optional[int] = 1,
    timeout: Optional[float] = None,
    prefer_greedy: bool = False,
    overlap_ratio: float = 0.0,
    verbose: bool = True,
    missing_pattern: str = MCAR,
) -> ExperimentSummary:
    log = print if verbose else (lambda *args, **kwargs: None)
    effective_max_workers = max(1, max_workers or (os.cpu_count() or 1))
    effective_parallel_threshold = max(1, parallel_threshold)
    configure_parallel_runtime(
        effective_max_workers,
        effective_parallel_threshold,
        chunk_size,
    )

    feature_names, feature_rows, label_names, label_rows = read_arff(data_path)
    features = np.asarray(feature_rows, dtype=np.float64)
    labels = np.asarray(label_rows, dtype=np.int8)

    original_object_count = features.shape[0]
    features, labels = _sample_dataset(
        features,
        labels,
        sample_fraction=sample_fraction,
        max_objects=max_objects,
        seed=sample_seed if sample_seed is not None else seed,
    )
    sampled_object_count = features.shape[0]

    features_missing, injected_count = inject_missing_values(
        features,
        missing_ratio,
        seed,
        pattern=missing_pattern,
    )
    folds_indices = build_kfold_indices(
        sampled_object_count,
        folds,
        seed,
        overlap_ratio=overlap_ratio,
    )

    chunk_desc = str(chunk_size) if chunk_size is not None else "auto"
    header = title or f"{algorithm} experiment"
    log(f"=== {header} ===")
    log(f"Dataset path           : {data_path}")
    log(f"Original objects       : {original_object_count}")
    if sampled_object_count != original_object_count:
        log(
            f"Sampled objects        : {sampled_object_count} "
            f"({sampled_object_count / original_object_count:.2%} of original)"
        )
    log(f"Objects                : {sampled_object_count}")
    log(f"Original attributes    : {len(feature_names)}")
    log(f"Labels                 : {len(label_names)}")
    log(f"Algorithm              : {algorithm}")
    log(f"Missing ratio target   : {missing_ratio:.2%}")
    log(f"Missing pattern        : {missing_pattern}")
    log(f"Missing cells injected : {injected_count}")
    log(f"Folds                  : {folds}")
    log(f"Train/Test overlap     : 0.00% (standard K-fold)")
    log(f"Max workers            : {effective_max_workers}")
    log(f"Parallel threshold     : {effective_parallel_threshold}")
    log(f"Map chunk size         : {chunk_desc}")
    if max_reducts is not None:
        log(f"Max reducts per fold   : {max_reducts}")
    log(f"Prefer greedy reduct   : {'yes' if prefer_greedy else 'no'}")
    if timeout is not None:
        log(f"Reduction timeout (s)  : {timeout:.2f}")
    log("")

    fold_results: List[FoldResult] = []
    overall_start = time.perf_counter()
    for fold_idx, (train_indices, test_indices) in enumerate(folds_indices, start=1):
        log(f"--- Fold {fold_idx:02d}/{folds} ---")
        train_X_missing = features_missing[train_indices]
        train_y = labels[train_indices]
        test_X_missing = features_missing[test_indices]
        test_y = labels[test_indices]

        selection = _select_features_for_algorithm(
            algorithm=algorithm,
            train_X_missing=train_X_missing,
            train_y=train_y,
            feature_names=feature_names,
            label_names=label_names,
            seed=seed + fold_idx,
            max_reducts=max_reducts,
            timeout=timeout,
            prefer_greedy=prefer_greedy,
        )

        selected_train = train_X_missing[:, selection.selected_indices]
        selected_test = test_X_missing[:, selection.selected_indices]
        classifier_train, classifier_test = _prepare_classifier_data(
            algorithm=algorithm,
            imputer_name=selection.imputer,
            X_train=selected_train,
            X_test=selected_test,
            seed=seed + fold_idx,
        )
        metrics = _evaluate_with_mlknn(
            classifier_train,
            train_y,
            classifier_test,
            test_y,
            label_names,
        )
        reduction_rate = (
            1.0 - len(selection.selected_indices) / max(1, len(feature_names))
        ) * 100.0

        validation_objects = max(1, int(round(len(train_indices) * 0.2))) if algorithm not in (OURS, MCB_AR) else 0
        fold_result = FoldResult(
            fold_index=fold_idx,
            algorithm=algorithm,
            imputer=selection.imputer,
            tuned_k=selection.tuned_k,
            train_objects=len(train_indices),
            validation_objects=validation_objects,
            test_objects=len(test_indices),
            reduction_time=selection.reduction_time,
            tuning_time=selection.tuning_time,
            preprocessing_time=selection.preprocessing_time,
            selection_pipeline_time=selection.selection_pipeline_time,
            selected_attributes=len(selection.selected_indices),
            reduction_rate=reduction_rate,
            test_f1_score=float(metrics["f1_score"]),
            test_ranking_loss=float(metrics["ranking_loss"]),
            test_coverage=float(metrics["coverage"]),
            test_hamming_loss=float(metrics["hamming_loss"]),
            chosen_features=selection.selected_names,
        )
        fold_results.append(fold_result)

        log(f"Train objects          : {fold_result.train_objects}")
        log(f"Validation objects     : {fold_result.validation_objects}")
        log(f"Test objects           : {fold_result.test_objects}")
        log(f"Reduction time (s)     : {fold_result.reduction_time:.6f}")
        log(f"Tuning time (s)        : {fold_result.tuning_time:.6f}")
        log(f"Prep time (s)          : {fold_result.preprocessing_time:.6f}")
        log(f"Selection pipeline (s) : {fold_result.selection_pipeline_time:.6f}")
        log(f"Selected attributes    : {fold_result.selected_attributes}")
        log(f"Reduction rate (%)     : {fold_result.reduction_rate:.2f}")
        log(f"Test F1-score          : {fold_result.test_f1_score:.6f}")
        log(f"Test Ranking Loss      : {fold_result.test_ranking_loss:.6f}")
        log(f"Test Coverage          : {fold_result.test_coverage:.6f}")
        log(f"Test Hamming Loss      : {fold_result.test_hamming_loss:.6f}")
        log(f"Selection imputer      : {fold_result.imputer}")
        log(f"Tuned k                : {fold_result.tuned_k}")
        log(f"Chosen features        : {', '.join(fold_result.chosen_features)}")
        log("")

    total_runtime = time.perf_counter() - overall_start
    avg_reduction_time, std_reduction_time = _mean_std(
        fr.reduction_time for fr in fold_results
    )
    avg_tuning_time, std_tuning_time = _mean_std(fr.tuning_time for fr in fold_results)
    avg_preprocessing_time, std_preprocessing_time = _mean_std(
        fr.preprocessing_time for fr in fold_results
    )
    avg_selection_pipeline_time, std_selection_pipeline_time = _mean_std(
        fr.selection_pipeline_time for fr in fold_results
    )
    avg_selected_attributes, std_selected_attributes = _mean_std(
        fr.selected_attributes for fr in fold_results
    )
    avg_reduction_rate, std_reduction_rate = _mean_std(
        fr.reduction_rate for fr in fold_results
    )
    avg_f1_score, std_f1_score = _mean_std(fr.test_f1_score for fr in fold_results)
    avg_ranking_loss, std_ranking_loss = _mean_std(
        fr.test_ranking_loss for fr in fold_results
    )
    avg_coverage, std_coverage = _mean_std(fr.test_coverage for fr in fold_results)
    avg_hamming_loss, std_hamming_loss = _mean_std(
        fr.test_hamming_loss for fr in fold_results
    )

    log("=== Summary ===")
    log(f"Average reduction time (s) : {avg_reduction_time:.6f} ± {std_reduction_time:.6f}")
    log(f"Average tuning time (s)    : {avg_tuning_time:.6f} ± {std_tuning_time:.6f}")
    log(
        f"Average prep time (s)      : "
        f"{avg_preprocessing_time:.6f} ± {std_preprocessing_time:.6f}"
    )
    log(
        f"Average pipeline time (s)  : "
        f"{avg_selection_pipeline_time:.6f} ± {std_selection_pipeline_time:.6f}"
    )
    log(f"Average selected attributes: {avg_selected_attributes:.2f} ± {std_selected_attributes:.2f}")
    log(f"Average reduction rate (%) : {avg_reduction_rate:.2f} ± {std_reduction_rate:.2f}")
    log(f"Average F1-score           : {avg_f1_score:.6f} ± {std_f1_score:.6f}")
    log(f"Average Ranking Loss       : {avg_ranking_loss:.6f} ± {std_ranking_loss:.6f}")
    log(f"Average Coverage           : {avg_coverage:.6f} ± {std_coverage:.6f}")
    log(f"Average Hamming Loss       : {avg_hamming_loss:.6f} ± {std_hamming_loss:.6f}")
    log(f"Total wall time (s)        : {total_runtime:.6f}")

    dominant_imputer = (
        fold_results[0].imputer
        if all(fr.imputer == fold_results[0].imputer for fr in fold_results)
        else "mixed"
    )
    return ExperimentSummary(
        dataset_path=data_path,
        algorithm=algorithm,
        imputer=dominant_imputer,
        missing_ratio=missing_ratio,
        missing_pattern=missing_pattern,
        fold_results=fold_results,
        avg_reduction_time=avg_reduction_time,
        std_reduction_time=std_reduction_time,
        avg_tuning_time=avg_tuning_time,
        std_tuning_time=std_tuning_time,
        avg_preprocessing_time=avg_preprocessing_time,
        std_preprocessing_time=std_preprocessing_time,
        avg_selection_pipeline_time=avg_selection_pipeline_time,
        std_selection_pipeline_time=std_selection_pipeline_time,
        avg_selected_attributes=avg_selected_attributes,
        std_selected_attributes=std_selected_attributes,
        avg_reduction_rate=avg_reduction_rate,
        std_reduction_rate=std_reduction_rate,
        avg_f1_score=avg_f1_score,
        std_f1_score=std_f1_score,
        avg_ranking_loss=avg_ranking_loss,
        std_ranking_loss=std_ranking_loss,
        avg_coverage=avg_coverage,
        std_coverage=std_coverage,
        avg_hamming_loss=avg_hamming_loss,
        std_hamming_loss=std_hamming_loss,
        total_runtime=total_runtime,
    )


def run_experiment(**kwargs) -> ExperimentSummary:
    return run_algorithm_experiment(**kwargs)


def run_comparison_experiment(
    *,
    data_path: Path,
    folds: int = DEFAULT_FOLDS,
    missing_ratio: float = MISSING_RATIO,
    seed: int = RANDOM_SEED,
    algorithms: Sequence[str] = (OURS, MFS_MCDM, FIMF, ML_CSFS),
    max_workers: Optional[int] = None,
    parallel_threshold: int = 1,
    chunk_size: Optional[int] = None,
    sample_fraction: float = 1.0,
    max_objects: Optional[int] = None,
    sample_seed: Optional[int] = None,
    max_reducts: Optional[int] = 1,
    timeout: Optional[float] = None,
    prefer_greedy: bool = False,
    verbose: bool = True,
    missing_pattern: str = MCAR,
) -> ComparisonResult:
    summaries: Dict[str, ExperimentSummary] = {}
    dataset_name = data_path.stem
    for algorithm in algorithms:
        summaries[algorithm] = run_algorithm_experiment(
            data_path=data_path,
            algorithm=algorithm,
            folds=folds,
            missing_ratio=missing_ratio,
            seed=seed,
            max_workers=max_workers,
            parallel_threshold=parallel_threshold,
            chunk_size=chunk_size,
            title=f"{dataset_name} | {algorithm} | Missing {missing_ratio:.0%}",
            sample_fraction=sample_fraction,
            max_objects=max_objects,
            sample_seed=sample_seed,
            max_reducts=max_reducts,
            timeout=timeout,
            prefer_greedy=(prefer_greedy if algorithm == OURS else False),
            overlap_ratio=0.0,
            verbose=verbose,
            missing_pattern=missing_pattern,
        )
    return ComparisonResult(
        dataset_name=dataset_name,
        missing_ratio=missing_ratio,
        missing_pattern=missing_pattern,
        summaries=summaries,
    )


def compute_wilcoxon_significance(
    task_results: Sequence[ComparisonResult],
    *,
    baseline_algorithms: Sequence[str] = BASELINE_ALGORITHMS,
) -> List[SignificanceRecord]:
    metrics = (
        ("f1_score", "avg_f1_score"),
        ("ranking_loss", "avg_ranking_loss"),
        ("coverage", "avg_coverage"),
        ("hamming_loss", "avg_hamming_loss"),
        ("reduction_time", "avg_reduction_time"),
    )
    records: List[SignificanceRecord] = []

    for metric_name, attr_name in metrics:
        ours_values = [
            getattr(task.summaries[OURS], attr_name)
            for task in task_results
            if OURS in task.summaries
        ]
        if len(ours_values) < 2:
            continue

        for baseline in baseline_algorithms:
            baseline_values = [
                getattr(task.summaries[baseline], attr_name)
                for task in task_results
                if baseline in task.summaries
            ]
            if len(baseline_values) != len(ours_values) or len(baseline_values) < 2:
                continue

            try:
                statistic, p_value = wilcoxon(ours_values, baseline_values)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            records.append(
                SignificanceRecord(
                    metric=metric_name,
                    baseline=baseline,
                    statistic=float(statistic),
                    p_value=float(p_value),
                    ours_mean=float(statistics.mean(ours_values)),
                    baseline_mean=float(statistics.mean(baseline_values)),
                )
            )
    return records
