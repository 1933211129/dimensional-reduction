from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


DEFAULT_LAMBDA = 10.0


@dataclass(frozen=True)
class TopsisResult:
    """
    TOPSIS 特征排序结果。

    Attributes
    ----------
    feature_names:
        原始特征名称列表。
    scores:
        相对贴近度得分数组，索引与 feature_names 对齐。
    ranking_indices:
        按得分从大到小排序后的特征索引。
    label_weights:
        标签熵权重向量。
    coefficients:
        岭回归得到的系数矩阵，形状为 (特征数, 标签数)。
    """

    feature_names: List[str]
    scores: np.ndarray
    ranking_indices: List[int]
    label_weights: np.ndarray
    coefficients: np.ndarray

    @property
    def ranking(self) -> List[str]:
        """按相对贴近度从高到低返回特征名称。"""
        return [self.feature_names[idx] for idx in self.ranking_indices]

    def score_map(self) -> Dict[str, float]:
        """以字典形式返回特征得分。"""
        return {
            feature: float(self.scores[idx])
            for idx, feature in enumerate(self.feature_names)
        }


def topsis_feature_ranking(
    *,
    feature_names: Sequence[str],
    feature_rows: Sequence[Sequence[object]],
    label_rows: Sequence[Sequence[object]],
    missing_token: object = "*",
    lambda_reg: float = DEFAULT_LAMBDA,
    impute_strategy: str = "mean",
) -> TopsisResult:
    """
    根据 TOPSIS（结合熵权与岭回归）计算特征排序。

    Parameters
    ----------
    feature_names:
        特征名称列表，长度等于特征维度。
    feature_rows:
        样本的特征取值，每行对应一个对象。
    label_rows:
        样本的标签取值（多标签），与 feature_rows 行数一致。
    missing_token:
        缺失值占位符，将在数值化阶段替换为 NaN。
    lambda_reg:
        岭回归正则化系数 λ。
    impute_strategy:
        缺失值填补策略，可选 "mean"、"median"、"zero"。

    Returns
    -------
    TopsisResult
        含特征得分、排序、熵权与系数矩阵。
    """

    if not feature_names:
        raise ValueError("feature_names 不能为空。")
    if not feature_rows:
        raise ValueError("feature_rows 不能为空。")
    if not label_rows:
        raise ValueError("label_rows 不能为空。")

    object_count = len(feature_rows)
    if len(label_rows) != object_count:
        raise ValueError("特征与标签的样本数量不一致。")

    feature_matrix = _features_to_matrix(feature_rows, missing_token=missing_token)
    imputed_matrix = _impute_missing_values(feature_matrix, strategy=impute_strategy)
    label_matrix = _labels_to_matrix(label_rows)

    if label_matrix.shape[0] != object_count:
        raise ValueError("标签矩阵的行数必须与样本数量一致。")

    label_weights = _entropy_weights(label_matrix)
    coefficients = _ridge_regression_coefficients(
        imputed_matrix,
        label_matrix,
        lambda_reg=lambda_reg,
    )
    scores = _relative_closeness(coefficients, label_weights)

    ranking_indices = np.argsort(-scores, kind="mergesort")
    return TopsisResult(
        feature_names=list(feature_names),
        scores=scores,
        ranking_indices=ranking_indices.tolist(),
        label_weights=label_weights,
        coefficients=coefficients,
    )


def _features_to_matrix(
    rows: Sequence[Sequence[object]],
    *,
    missing_token: object = "*",
) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=float)

    column_count = len(rows[0])
    matrix = np.empty((len(rows), column_count), dtype=float)
    matrix.fill(np.nan)

    for i, row in enumerate(rows):
        if len(row) != column_count:
            raise ValueError("所有样本的特征维度必须一致。")
        for j, value in enumerate(row):
            matrix[i, j] = _to_float(value, missing_token=missing_token)
    return matrix


def _labels_to_matrix(rows: Sequence[Sequence[object]]) -> np.ndarray:
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("标签矩阵必须是二维的。")
    return matrix


def _to_float(value: object, *, missing_token: object) -> float:
    if value is None:
        return math.nan
    if missing_token is not None and value == missing_token:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return math.nan
        if missing_token is not None and stripped == str(missing_token):
            return math.nan
        if stripped == "?":
            return math.nan
        return float(stripped)
    raise TypeError(f"无法将类型 {type(value)} 转为浮点数。")


def _impute_missing_values(matrix: np.ndarray, *, strategy: str) -> np.ndarray:
    if matrix.size == 0:
        return matrix.copy()

    strategy_lower = strategy.lower()
    if strategy_lower not in {"mean", "median", "zero"}:
        raise ValueError('strategy 必须是 "mean"、"median" 或 "zero"。')

    imputed = matrix.copy()
    if strategy_lower == "zero":
        replacement = np.zeros(imputed.shape[1], dtype=float)
    elif strategy_lower == "median":
        replacement = np.nanmedian(imputed, axis=0)
    else:
        replacement = np.nanmean(imputed, axis=0)

    replacement = np.where(np.isnan(replacement), 0.0, replacement)
    missing_indices = np.where(np.isnan(imputed))
    if missing_indices[0].size > 0:
        imputed[missing_indices] = np.take(replacement, missing_indices[1])
    return imputed


def _entropy_weights(label_matrix: np.ndarray) -> np.ndarray:
    if label_matrix.size == 0:
        return np.empty(0, dtype=float)

    entropies = []
    for column in label_matrix.T:
        entropies.append(_shannon_entropy(column))
    entropies_array = np.asarray(entropies, dtype=float)

    total = entropies_array.sum()
    if total <= 0:
        # 当所有标签全为 0/1 时熵为 0，退化为等权。
        return np.full_like(entropies_array, 1.0 / len(entropies_array))
    return entropies_array / total


def _shannon_entropy(column: np.ndarray) -> float:
    values, counts = np.unique(column, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(probabilities)
    logs[~np.isfinite(logs)] = 0.0
    return float(-(probabilities * logs).sum())


def _ridge_regression_coefficients(
    feature_matrix: np.ndarray,
    label_matrix: np.ndarray,
    *,
    lambda_reg: float,
) -> np.ndarray:
    if feature_matrix.ndim != 2 or label_matrix.ndim != 2:
        raise ValueError("特征矩阵与标签矩阵必须是二维的。")
    if feature_matrix.shape[0] != label_matrix.shape[0]:
        raise ValueError("特征矩阵与标签矩阵的样本数量不一致。")

    _, feature_count = feature_matrix.shape
    if feature_count == 0:
        return np.zeros((0, label_matrix.shape[1]), dtype=float)

    xtx = feature_matrix.T @ feature_matrix
    ridge_matrix = xtx + lambda_reg * np.eye(feature_count, dtype=float)
    xty = feature_matrix.T @ label_matrix

    try:
        coefficients = np.linalg.solve(ridge_matrix, xty)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(ridge_matrix) @ xty
    return coefficients


def _relative_closeness(
    coefficients: np.ndarray,
    label_weights: np.ndarray,
) -> np.ndarray:
    if coefficients.ndim != 2:
        raise ValueError("系数矩阵必须是二维的。")
    if coefficients.shape[1] != label_weights.shape[0]:
        raise ValueError("标签权重数量必须与系数矩阵的列数一致。")

    if coefficients.size == 0:
        return np.zeros((coefficients.shape[0],), dtype=float)

    weighted = coefficients * label_weights[np.newaxis, :]
    positive = np.max(weighted, axis=0, keepdims=True)
    negative = np.min(weighted, axis=0, keepdims=True)

    s_plus = np.sqrt(np.sum((weighted - positive) ** 2, axis=1))
    s_minus = np.sqrt(np.sum((weighted - negative) ** 2, axis=1))

    denominator = s_plus + s_minus
    scores = np.zeros_like(s_plus)
    valid = denominator > 0
    scores[valid] = s_minus[valid] / denominator[valid]
    return scores

