"""多标签分类与特征选择评估指标。"""

from __future__ import annotations

from typing import Sequence, Set

import numpy as np

__all__ = [
    "hamming_loss",
    "coverage",
    "reduction_rate",
    "ranking_loss",
    "f1_score_macro",
]


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 与 y_pred 形状必须一致。")
    mismatches = np.not_equal(y_true, y_pred).sum()
    return mismatches / (y_true.shape[0] * y_true.shape[1])


def coverage(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true 与 y_score 形状必须一致。")

    num_samples, num_labels = y_true.shape
    if num_labels == 0:
        return 0.0

    ranks = np.argsort(-y_score, axis=1)  # 概率从大到小排序后的标签索引
    coverage_sum = 0.0
    for i in range(num_samples):
        positives = np.flatnonzero(y_true[i])
        if positives.size == 0:
            continue
        rank_positions = np.empty(num_labels, dtype=int)
        rank_positions[ranks[i]] = np.arange(1, num_labels + 1)
        max_rank = rank_positions[positives].max()
        coverage_sum += max_rank - 1

    return coverage_sum / num_samples


def reduction_rate(num_selected: int, num_total: int) -> float:
    if num_total <= 0:
        raise ValueError("原始特征数必须为正。")
    if num_selected < 0 or num_selected > num_total:
        raise ValueError("num_selected 取值非法。")
    return 1.0 - num_selected / num_total


def ranking_loss(
    score_dicts: Sequence[dict[str, float]],
    truths: Sequence[Set[str]],
) -> float:
    if len(score_dicts) != len(truths):
        raise ValueError(
            f"得分字典序列长度 ({len(score_dicts)}) 与真实标签序列长度 ({len(truths)}) 不一致。"
        )

    if not score_dicts:
        return 0.0

    total_loss = 0.0
    valid_samples = 0

    for scores, truth_labels in zip(score_dicts, truths):
        relevant_labels = set(truth_labels)
        all_labels = set(scores.keys())
        irrelevant_labels = all_labels - relevant_labels

        if not relevant_labels or not irrelevant_labels:
            continue

        valid_samples += 1
        errors = 0

        for label in relevant_labels:
            if label not in scores:
                for label_prime in irrelevant_labels:
                    if label_prime in scores:
                        errors += 1
                continue

            score_label = scores[label]
            for label_prime in irrelevant_labels:
                if label_prime not in scores:
                    continue
                score_prime = scores[label_prime]
                if score_label <= score_prime:
                    errors += 1

        denominator = len(relevant_labels) * len(irrelevant_labels)
        if denominator > 0:
            total_loss += errors / denominator

    if valid_samples == 0:
        return 0.0

    return total_loss / valid_samples


def f1_score_macro(
    predictions: Sequence[Set[str]],
    truths: Sequence[Set[str]],
) -> float:
    if len(predictions) != len(truths):
        raise ValueError(
            f"预测序列长度 ({len(predictions)}) 与真实标签序列长度 ({len(truths)}) 不一致。"
        )

    if not predictions:
        return 0.0

    pred_matrix, truth_matrix = _prepare_matrices(predictions, truths)

    if pred_matrix.size == 0 and truth_matrix.size == 0:
        return 1.0

    intersection = (pred_matrix * truth_matrix).sum(axis=1)
    pred_sizes = pred_matrix.sum(axis=1)
    truth_sizes = truth_matrix.sum(axis=1)

    denominator = pred_sizes + truth_sizes
    both_empty = (pred_sizes == 0) & (truth_sizes == 0)
    one_empty = (pred_sizes == 0) | (truth_sizes == 0)

    f1_scores = np.zeros_like(denominator, dtype=np.float32)
    f1_scores[both_empty] = 1.0
    valid_mask = ~one_empty
    f1_scores[valid_mask] = (2.0 * intersection[valid_mask]) / denominator[valid_mask]

    return float(f1_scores.mean())


def _prepare_matrices(
    predictions: Sequence[Set[str]],
    truths: Sequence[Set[str]],
) -> tuple[np.ndarray, np.ndarray]:
    if not predictions:
        return np.empty((0, 0), dtype=np.int8), np.empty((0, 0), dtype=np.int8)

    label_space: Set[str] = set()
    for pred in predictions:
        label_space.update(pred)
    for truth in truths:
        label_space.update(truth)

    if not label_space:
        num_samples = len(predictions)
        return (
            np.zeros((num_samples, 0), dtype=np.int8),
            np.zeros((num_samples, 0), dtype=np.int8),
        )

    labels = sorted(label_space)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    num_samples = len(predictions)
    num_labels = len(labels)

    pred_matrix = np.zeros((num_samples, num_labels), dtype=np.int8)
    truth_matrix = np.zeros((num_samples, num_labels), dtype=np.int8)

    for idx, (pred_set, truth_set) in enumerate(zip(predictions, truths)):
        for label in pred_set:
            label_idx = label_to_index.get(label)
            if label_idx is not None:
                pred_matrix[idx, label_idx] = 1
        for label in truth_set:
            label_idx = label_to_index.get(label)
            if label_idx is not None:
                truth_matrix[idx, label_idx] = 1

    return pred_matrix, truth_matrix

