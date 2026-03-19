"""
多标记分类评价指标模块

实现以下评价指标：
1. 单个样本的精确率 (Precision_t)
2. 单个样本的召回率 (Recall_t)
3. 单个样本的F1-score (F1score_t)
4. 整体多标记F1-score (F1score)
5. One Error (OneError)
6. Ranking Loss (RankLoss)
7. Coverage (覆盖率)
8. Hamming Loss (汉明损失)

使用numpy进行向量化计算以提升性能。
"""

from __future__ import annotations

from typing import Sequence, Set

import numpy as np


def precision_per_sample(
    predicted: Set[str],
    truth: Set[str],
) -> float:
    """
    计算单个样本的精确率 (Precision_t)。

    定义: 预测的标签子集中，真实标签的比例。
    公式: Precision_t = |Z_t ∩ Y_t| / |Z_t|

    Args:
        predicted: 预测的标签集合 Z_t
        truth: 真实的标签集合 Y_t

    Returns:
        单个样本的精确率，范围 [0, 1]。
        如果预测集为空，返回 0.0（避免除零错误）。
    """
    if not predicted:
        return 0.0
    intersection_size = len(predicted & truth)
    return intersection_size / len(predicted)


def recall_per_sample(
    predicted: Set[str],
    truth: Set[str],
) -> float:
    """
    计算单个样本的召回率 (Recall_t)。

    定义: 真实标签子集中，被正确预测的比例。
    公式: Recall_t = |Z_t ∩ Y_t| / |Y_t|

    Args:
        predicted: 预测的标签集合 Z_t
        truth: 真实的标签集合 Y_t

    Returns:
        单个样本的召回率，范围 [0, 1]。
        如果真实标签集为空，返回 0.0（避免除零错误）。
    """
    if not truth:
        return 0.0
    intersection_size = len(predicted & truth)
    return intersection_size / len(truth)


def f1_score_per_sample(
    predicted: Set[str],
    truth: Set[str],
) -> float:
    """
    计算单个样本的F1-score (F1score_t)。

    定义: 精确率与召回率的调和平均数。
    公式: F1score_t = 2 * |Z_t ∩ Y_t| / (|Z_t| + |Y_t|)

    Args:
        predicted: 预测的标签集合 Z_t
        truth: 真实的标签集合 Y_t

    Returns:
        单个样本的F1-score，范围 [0, 1]。
        如果预测集和真实标签集都为空，返回 1.0（完全匹配）。
        如果只有一方为空，返回 0.0。
    """
    intersection_size = len(predicted & truth)
    predicted_size = len(predicted)
    truth_size = len(truth)
    
    # 如果两者都为空，认为是完全匹配
    if predicted_size == 0 and truth_size == 0:
        return 1.0
    
    # 如果只有一方为空，F1为0
    if predicted_size == 0 or truth_size == 0:
        return 0.0
    
    # 标准F1公式：2 * intersection / (predicted + truth)
    denominator = predicted_size + truth_size
    if denominator == 0:
        return 0.0
    
    return (2.0 * intersection_size) / denominator


def _sets_to_binary_matrix(
    sets: Sequence[Set[str]],
    label_to_idx: dict[str, int],
) -> np.ndarray:
    """
    将标签集合序列转换为二进制矩阵。

    Args:
        sets: 标签集合序列
        label_to_idx: 标签到索引的映射

    Returns:
        形状为 (n_samples, n_labels) 的二进制矩阵
    """
    n_samples = len(sets)
    n_labels = len(label_to_idx)
    matrix = np.zeros((n_samples, n_labels), dtype=np.float32)
    
    for i, label_set in enumerate(sets):
        for label in label_set:
            if label in label_to_idx:
                matrix[i, label_to_idx[label]] = 1.0
    
    return matrix


def _prepare_matrices(
    predictions: Sequence[Set[str]],
    truths: Sequence[Set[str]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    将预测和真实标签集合序列转换为二进制矩阵。

    Args:
        predictions: 预测标签集合序列
        truths: 真实标签集合序列

    Returns:
        (pred_matrix, truth_matrix) 二元组，形状均为 (n_samples, n_labels)
    """
    # 收集所有标签
    all_labels = set()
    for pred in predictions:
        all_labels.update(pred)
    for truth in truths:
        all_labels.update(truth)
    
    # 创建标签到索引的映射
    label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
    
    # 转换为二进制矩阵
    pred_matrix = _sets_to_binary_matrix(predictions, label_to_idx)
    truth_matrix = _sets_to_binary_matrix(truths, label_to_idx)
    
    return pred_matrix, truth_matrix


def f1_score_macro(
    predictions: Sequence[Set[str]],
    truths: Sequence[Set[str]],
) -> float:
    """
    计算整体多标记F1-score (F1score)。

    定义: 所有样本的单样本F1-score的平均值。
    公式: F1score = (1/p) * Σ F1score_t

    使用numpy向量化计算以提升性能。

    Args:
        predictions: 预测标签集合序列 [Z_1, Z_2, ..., Z_p]
        truths: 真实标签集合序列 [Y_1, Y_2, ..., Y_p]

    Returns:
        整体多标记F1-score，范围 [0, 1]。

    Raises:
        ValueError: 如果预测和真实标签序列长度不一致。
    """
    if len(predictions) != len(truths):
        raise ValueError(
            f"预测序列长度 ({len(predictions)}) 与真实标签序列长度 ({len(truths)}) 不一致。"
        )
    
    if not predictions:
        return 0.0
    
    # 转换为二进制矩阵
    pred_matrix, truth_matrix = _prepare_matrices(predictions, truths)
    
    # 检查是否所有集合都为空
    if pred_matrix.sum() == 0 and truth_matrix.sum() == 0:
        return 1.0  # 所有集合都为空，完全匹配
    
    # 向量化计算：交集大小
    intersection = (pred_matrix * truth_matrix).sum(axis=1)  # (n_samples,)
    pred_sizes = pred_matrix.sum(axis=1)  # (n_samples,)
    truth_sizes = truth_matrix.sum(axis=1)  # (n_samples,)
    
    # 计算F1-score: 2 * intersection / (pred_size + truth_size)
    # 处理边界情况：两者都为空时F1=1，只有一方为空时F1=0
    denominator = pred_sizes + truth_sizes
    both_empty = (pred_sizes == 0) & (truth_sizes == 0)
    one_empty = (pred_sizes == 0) | (truth_sizes == 0)
    
    f1_scores = np.zeros_like(denominator, dtype=np.float32)
    f1_scores[both_empty] = 1.0
    valid_mask = ~one_empty
    f1_scores[valid_mask] = (2.0 * intersection[valid_mask]) / denominator[valid_mask]
    
    return float(f1_scores.mean())


def precision_macro(
    predictions: Sequence[Set[str]],
    truths: Sequence[Set[str]],
) -> float:
    """
    计算整体多标记精确率（宏平均）。

    定义: 所有样本的单样本精确率的平均值。

    使用numpy向量化计算以提升性能。

    Args:
        predictions: 预测标签集合序列
        truths: 真实标签集合序列

    Returns:
        整体多标记精确率，范围 [0, 1]。
    """
    if len(predictions) != len(truths):
        raise ValueError(
            f"预测序列长度 ({len(predictions)}) 与真实标签序列长度 ({len(truths)}) 不一致。"
        )
    
    if not predictions:
        return 0.0
    
    # 转换为二进制矩阵
    pred_matrix, truth_matrix = _prepare_matrices(predictions, truths)
    
    # 检查是否所有集合都为空
    if pred_matrix.sum() == 0 and truth_matrix.sum() == 0:
        return 1.0  # 所有集合都为空，完全匹配
    
    # 向量化计算：交集大小和预测集大小
    intersection = (pred_matrix * truth_matrix).sum(axis=1)  # (n_samples,)
    pred_sizes = pred_matrix.sum(axis=1)  # (n_samples,)
    
    # 计算精确率: intersection / pred_size
    # 处理边界情况：预测集为空时精确率为0
    precision_scores = np.zeros_like(pred_sizes, dtype=np.float32)
    valid_mask = pred_sizes > 0
    precision_scores[valid_mask] = intersection[valid_mask] / pred_sizes[valid_mask]
    
    return float(precision_scores.mean())


def recall_macro(
    predictions: Sequence[Set[str]],
    truths: Sequence[Set[str]],
) -> float:
    """
    计算整体多标记召回率（宏平均）。

    定义: 所有样本的单样本召回率的平均值。

    使用numpy向量化计算以提升性能。

    Args:
        predictions: 预测标签集合序列
        truths: 真实标签集合序列

    Returns:
        整体多标记召回率，范围 [0, 1]。
    """
    if len(predictions) != len(truths):
        raise ValueError(
            f"预测序列长度 ({len(predictions)}) 与真实标签序列长度 ({len(truths)}) 不一致。"
        )
    
    if not predictions:
        return 0.0
    
    # 转换为二进制矩阵
    pred_matrix, truth_matrix = _prepare_matrices(predictions, truths)
    
    # 检查是否所有集合都为空
    if pred_matrix.sum() == 0 and truth_matrix.sum() == 0:
        return 1.0  # 所有集合都为空，完全匹配
    
    # 向量化计算：交集大小和真实标签集大小
    intersection = (pred_matrix * truth_matrix).sum(axis=1)  # (n_samples,)
    truth_sizes = truth_matrix.sum(axis=1)  # (n_samples,)
    
    # 计算召回率: intersection / truth_size
    # 处理边界情况：真实标签集为空时召回率为0
    recall_scores = np.zeros_like(truth_sizes, dtype=np.float32)
    valid_mask = truth_sizes > 0
    recall_scores[valid_mask] = intersection[valid_mask] / truth_sizes[valid_mask]
    
    return float(recall_scores.mean())


def one_error(
    score_dicts: Sequence[dict[str, float]],
    truths: Sequence[Set[str]],
) -> float:
    """
    计算 One Error 指标。
    
    定义: 考察在对象标记排序序列中，序列最前端的标记不属于相关标记集合的情况。
    公式: OneError(f) = (1/N) * Σ I[argmax_{l in L} f(x_i, l) not in L_{x_i}]
    
    该指标取值越小，则学习系统的性能越好。取值范围 [0, 1]。
    
    Args:
        score_dicts: 预测得分字典序列，每个字典为 {标签: 得分}
        truths: 真实标签集合序列 [Y_1, Y_2, ..., Y_N]
    
    Returns:
        One Error 值，范围 [0, 1]。
        0 表示所有样本的最高得分标签都在真实标签集合中。
        1 表示所有样本的最高得分标签都不在真实标签集合中。
    
    Raises:
        ValueError: 如果得分字典序列和真实标签序列长度不一致。
    """
    if len(score_dicts) != len(truths):
        raise ValueError(
            f"得分字典序列长度 ({len(score_dicts)}) 与真实标签序列长度 ({len(truths)}) 不一致。"
        )
    
    if not score_dicts:
        return 0.0
    
    errors = 0
    for scores, truth_labels in zip(score_dicts, truths):
        if not scores:
            # 如果没有得分，认为出错
            errors += 1
            continue
        
        # 找到得分最高的标签
        max_label = max(scores, key=scores.get)
        
        # 检查该标签是否不在真实标签集合中
        if max_label not in truth_labels:
            errors += 1
    
    return errors / len(score_dicts)


def ranking_loss(
    score_dicts: Sequence[dict[str, float]],
    truths: Sequence[Set[str]],
) -> float:
    """
    计算 Ranking Loss 指标。
    
    定义: 考察对象标记排序序列中出现排序错误的情况，即不相关标记在排序序列中位于相关标记之前。
    公式: RankLoss(f) = (1/N) * Σ |{(l, l') | f(x_i, l) <= f(x_i, l'), (l, l') in L_{x_i} × L_bar_{x_i}}| / (|L_{x_i}| × |L_bar_{x_i}|)
    
    该指标取值越小，则学习系统的性能越好。取值范围 [0, 1]。
    
    Args:
        score_dicts: 预测得分字典序列，每个字典为 {标签: 得分}
        truths: 真实标签集合序列 [Y_1, Y_2, ..., Y_N]
    
    Returns:
        Ranking Loss 值，范围 [0, 1]。
        0 表示所有相关标签的得分都高于不相关标签。
        1 表示所有相关标签的得分都低于或等于不相关标签。
    
    Raises:
        ValueError: 如果得分字典序列和真实标签序列长度不一致。
    """
    if len(score_dicts) != len(truths):
        raise ValueError(
            f"得分字典序列长度 ({len(score_dicts)}) 与真实标签序列长度 ({len(truths)}) 不一致。"
        )
    
    if not score_dicts:
        return 0.0
    
    total_loss = 0.0
    valid_samples = 0
    
    for scores, truth_labels in zip(score_dicts, truths):
        # 相关标签集合 L_{x_i}
        relevant_labels = truth_labels
        
        # 不相关标签集合 L_bar_{x_i} = 所有标签 - 相关标签
        all_labels = set(scores.keys())
        irrelevant_labels = all_labels - relevant_labels
        
        # 如果相关标签集为空或不相关标签集为空，跳过（分母为0）
        if not relevant_labels or not irrelevant_labels:
            continue
        
        valid_samples += 1
        
        # 计算排序错误的对数
        # 排序错误：相关标签 l 的得分 <= 不相关标签 l' 的得分
        errors = 0
        for l in relevant_labels:
            if l not in scores:
                # 如果相关标签没有得分，认为所有有得分的不相关标签都排在它前面
                for l_prime in irrelevant_labels:
                    if l_prime in scores:
                        errors += 1
                continue
            
            score_l = scores[l]
            for l_prime in irrelevant_labels:
                if l_prime not in scores:
                    # 如果不相关标签没有得分，跳过（无法比较）
                    continue
                
                score_l_prime = scores[l_prime]
                if score_l <= score_l_prime:
                    errors += 1
        
        # 分母：所有可能的相关-不相关标签对数量
        denominator = len(relevant_labels) * len(irrelevant_labels)
        
        if denominator > 0:
            total_loss += errors / denominator
    
    # 如果所有样本都无效（相关标签集或非相关标签集为空），返回0
    if valid_samples == 0:
        return 0.0
    
    return total_loss / valid_samples


def coverage(
    score_dicts: Sequence[dict[str, float]],
    truths: Sequence[Set[str]],
) -> float:
    """
    计算 Coverage (覆盖率) 指标。
    
    定义: 衡量平均需要将排序列表从顶部向下移动多少位，才能覆盖样本的所有真实标签。
    公式: Cov(T) = (1/p) * Σ [max_{y in Y_i} r_{y_i}(x_i, y) - 1]
    
    其中 r_{y_i}(x_i, y) 是标签 y 在排序列表中的排名（从1开始）。
    
    该指标取值越小，则学习系统的性能越好。
    
    Args:
        score_dicts: 预测得分字典序列，每个字典为 {标签: 得分}
        truths: 真实标签集合序列 [Y_1, Y_2, ..., Y_p]
    
    Returns:
        Coverage 值，非负实数。
        0 表示所有真实标签都排在第一位。
    
    Raises:
        ValueError: 如果得分字典序列和真实标签序列长度不一致。
    """
    if len(score_dicts) != len(truths):
        raise ValueError(
            f"得分字典序列长度 ({len(score_dicts)}) 与真实标签序列长度 ({len(truths)}) 不一致。"
        )
    
    if not score_dicts:
        return 0.0
    
    total_coverage = 0.0
    valid_samples = 0
    
    for scores, truth_labels in zip(score_dicts, truths):
        if not truth_labels:
            # 如果真实标签集为空，跳过
            continue
        
        valid_samples += 1
        
        # 如果得分字典为空，所有真实标签都没有排名，使用最大排名
        if not scores:
            # 假设所有标签的排名都是无穷大，但这里我们使用标签总数
            # 实际上，如果得分字典为空，我们无法确定排名
            # 为了计算的连续性，我们假设所有标签的排名都是 len(truth_labels)
            max_rank = len(truth_labels)
        else:
            # 对得分进行排序，得到排名（从1开始，1表示最好）
            # 得分相同的标签具有相同的排名（取最小排名）
            sorted_labels = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # 构建标签到排名的映射
            label_to_rank: dict[str, int] = {}
            current_rank = 1
            prev_score = None
            
            for i, (label, score) in enumerate(sorted_labels):
                if i == 0:
                    # 第一个标签排名为1
                    label_to_rank[label] = 1
                else:
                    # 如果得分不同，排名递增
                    if prev_score is not None and abs(score - prev_score) > 1e-10:
                        current_rank = i + 1
                    # 得分相同则保持当前排名
                    label_to_rank[label] = current_rank
                prev_score = score
            
            # 找到真实标签中排名最大的（排名值最大，即排在最后面）
            # max_{y in Y_i} r_{y_i}(x_i, y) 表示真实标签中排名值最大的
            max_rank = 0
            for label in truth_labels:
                if label in label_to_rank:
                    rank = label_to_rank[label]
                    if rank > max_rank:
                        max_rank = rank
                else:
                    # 如果真实标签不在得分字典中，认为它的排名是所有标签数 + 1
                    max_rank = max(max_rank, len(scores) + 1)
            
            # 如果所有真实标签都不在得分字典中
            if max_rank == 0:
                max_rank = len(scores) + 1
        
        # Coverage = max_rank - 1
        total_coverage += max_rank - 1
    
    # 如果所有样本都无效（真实标签集为空），返回0
    if valid_samples == 0:
        return 0.0
    
    return total_coverage / valid_samples


def hamming_loss(
    predictions: Sequence[Set[str]],
    truths: Sequence[Set[str]],
) -> float:
    """
    计算 Hamming Loss (汉明损失) 指标。
    
    定义: 衡量被错误预测的样本-标签对的比例，包含漏报和误报。
    公式: HL(T) = (1/p) * Σ [|Y_i Δ Z_i| / |L|]
    
    其中 Δ 表示两个集合的对称差（异或），|L| 是标签总数。
    
    该指标取值越小，则学习系统的性能越好。取值范围 [0, 1]。
    
    Args:
        predictions: 预测标签集合序列 [Z_1, Z_2, ..., Z_p]
        truths: 真实标签集合序列 [Y_1, Y_2, ..., Y_p]
    
    Returns:
        Hamming Loss 值，范围 [0, 1]。
        0 表示所有预测都完全正确。
        1 表示所有预测都完全错误。
    
    Raises:
        ValueError: 如果预测和真实标签序列长度不一致。
    """
    if len(predictions) != len(truths):
        raise ValueError(
            f"预测序列长度 ({len(predictions)}) 与真实标签序列长度 ({len(truths)}) 不一致。"
        )
    
    if not predictions:
        return 0.0
    
    # 收集所有标签以确定标签总数 |L|
    all_labels = set()
    for pred in predictions:
        all_labels.update(pred)
    for truth in truths:
        all_labels.update(truth)
    
    n_labels = len(all_labels)
    if n_labels == 0:
        return 0.0
    
    # 计算每个样本的对称差大小
    total_loss = 0.0
    for pred, truth in zip(predictions, truths):
        # 对称差：在 pred 中但不在 truth 中，或在 truth 中但不在 pred 中
        symmetric_diff = (pred - truth) | (truth - pred)
        total_loss += len(symmetric_diff) / n_labels
    
    return total_loss / len(predictions)

