"""数据加载与预处理工具。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "load_arff_multi_label",
    "inject_missing_values",
    "minmax_scale",
    "mean_impute",
    "apply_mean_impute",
]


def load_arff_multi_label(
    file_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """读取多标签ARFF数据集。

    返回
    ----
    X : np.ndarray
        特征矩阵，缺失值以 np.nan 表示。
    y : np.ndarray
        多标签0/1矩阵。
    feature_names : List[str]
        特征名称列表。
    label_names : List[str]
        标签名称列表。
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件: {path}")

    attributes: List[Tuple[str, str]] = []
    data_rows: List[List[str]] = []
    in_data_section = False

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lower = line.lower()
            if lower.startswith("@relation"):
                continue

            if lower.startswith("@attribute"):
                parts = line.split(None, 2)
                if len(parts) < 3:
                    raise ValueError(f"无法解析属性定义: {line}")
                name = parts[1].strip().strip("'\"")
                type_part = parts[2].strip()
                attributes.append((name, type_part))
                continue

            if lower.startswith("@data"):
                in_data_section = True
                continue

            if in_data_section:
                values = [v.strip() for v in line.split(",")]
                if values and values[0]:
                    data_rows.append(values)

    if not attributes or not data_rows:
        raise ValueError(f"数据文件 {path} 不包含有效的属性或数据。")

    num_attributes = len(attributes)
    feature_indices: List[int] = []
    label_indices: List[int] = []
    feature_names: List[str] = []
    label_names: List[str] = []

    for idx, (name, type_part) in enumerate(attributes):
        type_lower = type_part.lower()
        if type_lower.startswith("{") and type_lower.endswith("}"):
            values = [
                token.strip().strip("'\"") for token in type_part[1:-1].split(",")
            ]
            if set(values) <= {"0", "1"}:
                label_indices.append(idx)
                label_names.append(name)
            else:
                feature_indices.append(idx)
                feature_names.append(name)
        else:
            feature_indices.append(idx)
            feature_names.append(name)

    if not label_indices:
        raise ValueError(f"数据文件 {path} 中未识别到多标签字段。")

    num_samples = len(data_rows)
    X = np.empty((num_samples, len(feature_indices)), dtype=np.float32)
    y = np.empty((num_samples, len(label_indices)), dtype=np.int8)

    for row_idx, row in enumerate(data_rows):
        if len(row) != num_attributes:
            raise ValueError(
                f"第 {row_idx} 行属性数量({len(row)})与定义数量({num_attributes})不一致。"
            )
        for pos, attr_idx in enumerate(feature_indices):
            value = row[attr_idx]
            if value == "?" or value == "":
                X[row_idx, pos] = np.nan
            else:
                try:
                    X[row_idx, pos] = float(value)
                except ValueError as exc:
                    raise ValueError(
                        f"无法将值 '{value}' 解析为浮点数 (行 {row_idx}, 列 {attributes[attr_idx][0]})"
                    ) from exc
        for pos, attr_idx in enumerate(label_indices):
            value = row[attr_idx]
            if value == "?" or value == "":
                raise ValueError(
                    f"标签列 {attributes[attr_idx][0]} 在第 {row_idx} 行存在缺失值。"
                )
            y[row_idx, pos] = int(float(value))

    return X, y, feature_names, label_names


def inject_missing_values(
    X: np.ndarray,
    rate: float,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """按给定比例随机注入缺失值（仅针对特征矩阵）。"""
    if not 0 <= rate < 1:
        raise ValueError("缺失率应位于 [0, 1) 区间。")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    mask = rng.random(size=X.shape) < rate
    X_missing = X.copy()
    X_missing[mask] = np.nan
    return X_missing


def minmax_scale(
    X: np.ndarray,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对特征做列级Min-Max归一化，保留缺失值位置。

    返回归一化后的矩阵以及对应的最小值、最大值。
    """
    lower, upper = feature_range
    if upper <= lower:
        raise ValueError("feature_range 上界必须大于下界。")

    col_min = np.nanmin(X, axis=0)
    col_max = np.nanmax(X, axis=0)

    # 若整列缺失，则退化为常量0
    nan_columns = np.isnan(col_min) | np.isnan(col_max)
    if np.any(nan_columns):
        col_min = col_min.copy()
        col_max = col_max.copy()
        col_min[nan_columns] = 0.0
        col_max[nan_columns] = 0.0

    scale = col_max - col_min
    zero_scale_mask = scale == 0
    scale[zero_scale_mask] = 1.0  # 防止除以0

    X_norm = (X - col_min) / scale
    X_norm = X_norm * (upper - lower) + lower

    # 保留缺失值
    X_norm[np.isnan(X)] = np.nan
    return X_norm, col_min, col_max


def mean_impute(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """使用列均值进行缺失值填补。"""
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)

    X_imputed = X.copy()
    indices = np.where(np.isnan(X_imputed))
    if indices[0].size > 0:
        X_imputed[indices] = col_means[indices[1]]
    return X_imputed, col_means


def apply_mean_impute(X: np.ndarray, col_means: Sequence[float]) -> np.ndarray:
    """根据给定的列均值填补缺失值。"""
    X_imputed = X.copy()
    indices = np.where(np.isnan(X_imputed))
    if indices[0].size > 0:
        col_means_arr = np.asarray(col_means)
        X_imputed[indices] = col_means_arr[indices[1]]
    return X_imputed

