from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, FrozenSet, Tuple
import numpy as np


MISSING_VALUE = "*"

_TOLERANCE_DATA: Optional[Sequence[Sequence[str]]] = None
_DISCERN_DATA: Optional[Sequence[Sequence[str]]] = None
_DISCERN_PHI_BY_OBJECT: Optional[Sequence[Sequence[FrozenSet[int]]]] = None
_DISCERN_CONDITION_ATTRIBUTES: Optional[Sequence[str]] = None
_DISCERN_COARSE_SETS: Optional[Sequence[FrozenSet[str]]] = None
_DISCERN_FINE_SETS: Optional[Sequence[FrozenSet[str]]] = None
_DISCERN_PHI_INDICES: Optional[Sequence[Sequence[int]]] = None
_DISCERN_BLOCK_MISSING_ALL: Optional[Sequence[Sequence[bool]]] = None
_DISCERN_BLOCK_OBSERVED: Optional[Sequence[Sequence[FrozenSet[str]]]] = None

HittingSeed = Tuple[FrozenSet[str], Tuple[FrozenSet[str], ...]]

def _tolerance_worker_initializer(data: Sequence[Sequence[str]]) -> None:
    global _TOLERANCE_DATA
    _TOLERANCE_DATA = data


def _compute_tolerance_row_worker(index: int) -> Tuple[int, List[int]]:
    if _TOLERANCE_DATA is None:
        raise RuntimeError("Tolerance worker not initialized.")

    data = _TOLERANCE_DATA
    n = len(data)
    row_i = data[index]
    result = [0] * n
    result[index] = 1

    for j in range(index + 1, n):
        result[j] = 1 if _is_tolerant(row_i, data[j]) else 0

    return index, result


def _should_parallelize_tolerance(n: int) -> bool:
    if n <= 1:
        return False

    disable = os.getenv("ALGORITHM1_DISABLE_PARALLEL")
    if disable and disable.strip().lower() in {"1", "true", "yes", "on"}:
        return False

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return False

    threshold_env = os.getenv("ALGORITHM1_PARALLEL_THRESHOLD")
    try:
        threshold = int(threshold_env) if threshold_env is not None else 256
    except ValueError:
        threshold = 256

    return n >= threshold


def _should_parallelize_discernibility(n: int) -> bool:
    if n <= 1:
        return False

    disable_global = os.getenv("ALGORITHM1_DISABLE_PARALLEL")
    if disable_global and disable_global.strip().lower() in {"1", "true", "yes", "on"}:
        return False

    disable_discern = os.getenv("ALGORITHM1_DISABLE_DISCERN_PARALLEL")
    if disable_discern and disable_discern.strip().lower() in {"1", "true", "yes", "on"}:
        return False

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return False

    threshold_env = os.getenv("ALGORITHM1_DISCERN_PARALLEL_THRESHOLD")
    if not threshold_env:
        threshold_env = os.getenv("ALGORITHM1_PARALLEL_THRESHOLD")
    try:
        threshold = int(threshold_env) if threshold_env is not None else 64
    except ValueError:
        threshold = 64

    return n >= max(1, threshold)


def _should_parallelize_hitting_sets(clause_count: int, literal_count: int) -> bool:
    if clause_count <= 1:
        return False

    disable_global = os.getenv("ALGORITHM1_DISABLE_PARALLEL")
    if disable_global and disable_global.strip().lower() in {"1", "true", "yes", "on"}:
        return False

    disable_hitting = os.getenv("ALGORITHM1_DISABLE_HITTING_PARALLEL")
    if disable_hitting and disable_hitting.strip().lower() in {"1", "true", "yes", "on"}:
        return False

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return False

    threshold_env = os.getenv("ALGORITHM1_HITTING_PARALLEL_THRESHOLD")
    try:
        threshold = int(threshold_env) if threshold_env is not None else 8
    except ValueError:
        threshold = 8
    threshold = max(2, threshold)

    literal_threshold_env = os.getenv("ALGORITHM1_HITTING_LITERAL_THRESHOLD")
    try:
        literal_threshold = (
            int(literal_threshold_env) if literal_threshold_env is not None else 128
        )
    except ValueError:
        literal_threshold = 128

    return clause_count >= threshold or literal_count >= max(0, literal_threshold)


def _resolve_map_chunk_size(n: int, max_workers: int) -> int:
    chunk_env = os.getenv("ALGORITHM1_MAP_CHUNK_SIZE")
    chunk_size: Optional[int]
    if chunk_env:
        try:
            chunk_size = int(chunk_env)
        except ValueError:
            chunk_size = None
        else:
            if chunk_size <= 0:
                chunk_size = None
    else:
        chunk_size = None

    if chunk_size is None:
        denominator = max(1, max_workers * 8)
        chunk_size = max(1, n // denominator)

    return chunk_size


def _resolve_parallel_worker_budget() -> int:
    max_workers_env = os.getenv("ALGORITHM1_MAX_WORKERS")
    cpu_count = os.cpu_count() or 1
    if max_workers_env:
        try:
            requested = int(max_workers_env)
        except ValueError:
            requested = cpu_count
        else:
            requested = max(1, requested)
        return min(requested, cpu_count)
    return cpu_count


def _resolve_parallel_worker_count(n: int) -> int:
    max_workers_env = os.getenv("ALGORITHM1_MAX_WORKERS")
    cpu_count = os.cpu_count() or 1
    if max_workers_env:
        try:
            requested = int(max_workers_env)
        except ValueError:
            requested = cpu_count
        else:
            requested = max(1, requested)
        return min(requested, n, cpu_count)
    return min(cpu_count, n)


def _fill_tolerance_matrix_serial(
    matrix: List[List[int]], data: Sequence[Sequence[str]]
) -> None:
    n = len(data)
    for i in range(n):
        matrix[i][i] = 1
        row_i = data[i]
        for j in range(i + 1, n):
            value = 1 if _is_tolerant(row_i, data[j]) else 0
            matrix[i][j] = value
            matrix[j][i] = value


def _fill_tolerance_matrix_parallel(
    matrix: List[List[int]],
    data: Sequence[Sequence[str]],
    max_workers: int,
) -> None:
    n = len(data)
    data_tuple = tuple(tuple(row) for row in data)
    for i in range(n):
        matrix[i][i] = 1

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_tolerance_worker_initializer,
        initargs=(data_tuple,),
    ) as executor:
        chunk_size = _resolve_map_chunk_size(n, max_workers)
        task_iterable = range(n)
        for index, partial_row in executor.map(
            _compute_tolerance_row_worker, task_iterable, chunksize=chunk_size
        ):
            row = matrix[index]
            for j in range(index + 1, n):
                value = partial_row[j]
                row[j] = value
                matrix[j][index] = value


def _compute_tolerance_matrix_numpy(
    data: Sequence[Sequence[str]],
) -> np.ndarray:
    if not data:
        return np.zeros((0, 0), dtype=np.uint8)

    data_array = np.asarray(data, dtype=object)
    if data_array.ndim != 2:
        data_array = np.array([list(row) for row in data], dtype=object)

    object_count, attribute_count = data_array.shape
    tolerance = np.ones((object_count, object_count), dtype=bool)

    if attribute_count == 0:
        np.fill_diagonal(tolerance, True)
        return tolerance.astype(np.uint8)

    missing_mask = data_array == MISSING_VALUE

    for attr_idx in range(attribute_count):
        column = data_array[:, attr_idx]
        missing_column = missing_mask[:, attr_idx]
        equal_matrix = np.equal.outer(column, column)

        if missing_column.any():
            tolerant_attr = equal_matrix | missing_column[:, None] | missing_column[None, :]
        else:
            tolerant_attr = equal_matrix

        tolerance &= tolerant_attr

        if not tolerance.any():
            break

    np.fill_diagonal(tolerance, True)
    return tolerance.astype(np.uint8)


@dataclass
class IncompleteDecisionTable:
    condition_attributes: Sequence[str]
    condition_values: Sequence[Sequence[str]]
    decision_values: Sequence[str]

    def __post_init__(self) -> None:
        if not self.condition_attributes:
            raise ValueError("条件属性名称列表不能为空。")

        object_count = len(self.condition_values)
        if object_count == 0:
            raise ValueError("决策信息系统必须包含至少一个对象。")

        if len(self.decision_values) != object_count:
            raise ValueError("条件属性数据与决策属性数据的行数不一致。")

        expected_columns = len(self.condition_attributes)
        for idx, row in enumerate(self.condition_values):
            if len(row) != expected_columns:
                raise ValueError(
                    f"第 {idx + 1} 行条件属性数量为 {len(row)}，与属性名称数量 "
                    f"{expected_columns} 不一致。"
                )

    @property
    def object_count(self) -> int:
        return len(self.condition_values)

    @property
    def attribute_count(self) -> int:
        return len(self.condition_attributes)


@dataclass
class IncompleteMultiLabelDecisionTable:
    condition_attributes: Sequence[str]
    condition_values: Sequence[Sequence[str]]
    label_names: Sequence[str]
    label_values: Sequence[Sequence[object]]

    def __post_init__(self) -> None:
        if not self.condition_attributes:
            raise ValueError("条件属性名称列表不能为空。")
        if not self.label_names:
            raise ValueError("标签名称列表不能为空。")

        object_count = len(self.condition_values)
        if object_count == 0:
            raise ValueError("决策信息系统必须包含至少一个对象。")

        if len(self.label_values) != object_count:
            raise ValueError("条件属性数据与标签数据的行数不一致。")

        expected_cond_columns = len(self.condition_attributes)
        expected_label_columns = len(self.label_names)

        for idx, row in enumerate(self.condition_values):
            if len(row) != expected_cond_columns:
                raise ValueError(
                    f"第 {idx + 1} 行条件属性数量为 {len(row)}，与属性名称数量 "
                    f"{expected_cond_columns} 不一致。"
                )

        for idx, row in enumerate(self.label_values):
            if len(row) != expected_label_columns:
                raise ValueError(
                    f"第 {idx + 1} 行标签数量为 {len(row)}，与标签名称数量 "
                    f"{expected_label_columns} 不一致。"
                )

    @property
    def object_count(self) -> int:
        return len(self.condition_values)

    @property
    def attribute_count(self) -> int:
        return len(self.condition_attributes)

    @property
    def label_count(self) -> int:
        return len(self.label_names)


def _is_tolerant(row_i: Sequence[str], row_j: Sequence[str]) -> bool:
    """
    根据容差关系 SIM(A) 的定义，判定两个对象在所有条件属性上是否不可区分。
    任一属性只要存在非缺失且取值不相等，则二者不可容差。
    """
    for value_i, value_j in zip(row_i, row_j):
        if (
            value_i != MISSING_VALUE
            and value_j != MISSING_VALUE
            and value_i != value_j
        ):
            return False
    return True


def compute_tolerance_matrix(data: Sequence[Sequence[str]]) -> np.ndarray:
    """
    计算容差关系矩阵 F_{S(A)}，矩阵元素取 0/1。
    返回 numpy.ndarray，dtype 为 np.uint8。
    """
    n = len(data)
    if n == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    try:
        return _compute_tolerance_matrix_numpy(data)
    except MemoryError:
        pass
    except Exception:
        # 保持与旧实现一致的回退语义
        pass

    matrix = [[0] * n for _ in range(n)]

    if _should_parallelize_tolerance(n):
        max_workers = _resolve_parallel_worker_count(n)
        try:
            _fill_tolerance_matrix_parallel(matrix, data, max_workers)
        except Exception:
            matrix = [[0] * n for _ in range(n)]
            _fill_tolerance_matrix_serial(matrix, data)
        else:
            return np.asarray(matrix, dtype=np.uint8)
    else:
        _fill_tolerance_matrix_serial(matrix, data)

    return np.asarray(matrix, dtype=np.uint8)


def compute_tolerance_classes(
    tolerance_matrix: Sequence[Sequence[int]],
) -> List[Set[int]]:
    """
    根据容差关系矩阵计算每个对象的容差类 S_A(x_i)。
    返回的索引使用 0 开始的整数，方便后续运算。
    """
    tolerance_array = np.asarray(tolerance_matrix, dtype=np.uint8)
    classes: List[Set[int]] = []
    for row in tolerance_array:
        compatible_indices = np.flatnonzero(row).tolist()
        classes.append(set(compatible_indices))
    return classes


def compute_intersection_matrix(
    tolerance_classes: Sequence[Set[int]],
) -> List[List[Set[int]]]:
    """
    计算容差类交矩阵 IF_{S(A)}。
    若 j ∈ S_A(x_i)，则 IF_{ij} = S_A(x_i) ∩ S_A(x_j)，否则为空集。
    优化版本：使用更高效的集合交集操作。
    """
    n = len(tolerance_classes)
    intersection_matrix: List[List[Set[int]]] = [
        [set() for _ in range(n)] for _ in range(n)
    ]
    
    for i in range(n):
        s_i = tolerance_classes[i]
        # 如果s_i很小，直接遍历；如果很大，可以考虑优化
        for j in s_i:
            s_j = tolerance_classes[j]
            # 使用较小的集合进行交集，减少操作次数
            if len(s_i) <= len(s_j):
                intersection_matrix[i][j] = s_i & s_j
            else:
                intersection_matrix[i][j] = s_j & s_i
    return intersection_matrix


def _unique_frozensets(sets: Iterable[Iterable[int]]) -> List[FrozenSet[int]]:
    """
    对候选集合去重，并以确定性顺序输出（按元素升序、长度升序排列）。
    """
    unique = {frozenset(s) for s in sets if s}
    return sorted(unique, key=lambda block: (len(block), sorted(block)))


def _maximal_sets(candidates: Iterable[FrozenSet[int]]) -> List[FrozenSet[int]]:
    """
    从候选集合中过滤得到极大集合。
    若集合 A 是集合 B 的真子集，则只保留 B。
    优化版本：按长度降序排序，使用集合快速检查子集关系。
    """
    blocks = sorted(candidates, key=lambda block: (-len(block), sorted(block)))
    if not blocks:
        return []
    
    maximal: List[FrozenSet[int]] = []
    # 对于每个候选块，只需要检查是否被已有的极大集合包含
    for block in blocks:
        is_subset = False
        # 由于已按长度降序排序，只需要检查长度大于等于当前块的极大集合
        for other in maximal:
            if len(other) >= len(block) and block < other:
                is_subset = True
                break
        if not is_subset:
            maximal.append(block)
    
    # 为了便于阅读，最终再按长度升序、字典序排序
    return sorted(maximal, key=lambda block: (len(block), sorted(block)))


def _is_compatibility_block(
    block: Iterable[int], tolerance_matrix: Sequence[Sequence[int]]
) -> bool:
    """
    检查给定的对象下标集合是否构成相容块：任意两对象彼此容差。
    优化版本：对于小块直接检查，对于大块使用向量化操作。
    """
    indices = tuple(block)
    if len(indices) <= 1:
        return True
    
    if len(indices) == 2:
        # 小块直接检查，避免numpy开销
        i, j = indices
        tolerance_array = (
            tolerance_matrix
            if isinstance(tolerance_matrix, np.ndarray)
            else np.asarray(tolerance_matrix)
        )
        return bool(tolerance_array[i, j] and tolerance_array[j, i])

    tolerance_array = (
        tolerance_matrix
        if isinstance(tolerance_matrix, np.ndarray)
        else np.asarray(tolerance_matrix)
    )
    # 使用高级索引提取子矩阵
    submatrix = tolerance_array[np.ix_(indices, indices)]
    return np.all(submatrix == 1)


def _to_zero_based_blocks(blocks: Sequence[FrozenSet[int]]) -> List[FrozenSet[int]]:
    return [frozenset(idx - 1 for idx in block) for block in blocks]


def _compute_phi_by_object(
    blocks_zero_based: Sequence[FrozenSet[int]], object_count: int
) -> List[List[FrozenSet[int]]]:
    phi_by_object: List[List[FrozenSet[int]]] = [[] for _ in range(object_count)]
    for block in blocks_zero_based:
        for obj_idx in block:
            phi_by_object[obj_idx].append(block)
    return phi_by_object


def _build_block_attribute_cache(
    data: Sequence[Sequence[str]],
    blocks_zero_based: Sequence[FrozenSet[int]],
    attribute_count: int,
) -> Tuple[List[List[bool]], List[List[FrozenSet[str]]]]:
    """
    预先计算每个极大相容块在每个属性上的缺失情况与取值集合，
    以便在辨识子句构建时快速复用，避免重复遍历原始数据。
    """
    block_missing_all: List[List[bool]] = []
    block_observed_values: List[List[FrozenSet[str]]] = []

    for block in blocks_zero_based:
        missing_flags: List[bool] = []
        observed_lists: List[FrozenSet[str]] = []
        for attr_idx in range(attribute_count):
            observed = {
                data[obj_idx][attr_idx]
                for obj_idx in block
                if data[obj_idx][attr_idx] != MISSING_VALUE
            }
            missing_flags.append(len(observed) == 0)
            observed_lists.append(frozenset(observed))
        block_missing_all.append(missing_flags)
        block_observed_values.append(observed_lists)

    return block_missing_all, block_observed_values


def _build_object_attribute_codes(
    phi_block_indices_by_object: Sequence[Sequence[int]],
    block_missing_all: Sequence[Sequence[bool]],
    block_observed_values: Sequence[Sequence[FrozenSet[str]]],
    attribute_count: int,
) -> np.ndarray:
    """
    为每个对象在每个属性上构造紧凑编码：

    - 0 表示该属性在对象关联的块族中存在缺失块，或不同块的观测签名不一致；
    - 正整数表示该属性在对象关联的所有块中都具有同一个非缺失观测签名。

    这样一来，两个对象在某个属性上“不可辨识”的充要条件就是：
    二者在该属性上的编码相同且均为正数。
    """
    object_count = len(phi_block_indices_by_object)
    object_attr_codes = np.zeros((object_count, attribute_count), dtype=np.int32)
    signature_maps: List[Dict[FrozenSet[str], int]] = [
        {} for _ in range(attribute_count)
    ]

    for obj_idx, block_indices in enumerate(phi_block_indices_by_object):
        if not block_indices:
            continue

        for attr_idx in range(attribute_count):
            uniform_signature: Optional[FrozenSet[str]] = None
            is_uniform = True

            for block_idx in block_indices:
                if block_missing_all[block_idx][attr_idx]:
                    is_uniform = False
                    break

                observed_signature = block_observed_values[block_idx][attr_idx]
                if uniform_signature is None:
                    uniform_signature = observed_signature
                elif uniform_signature != observed_signature:
                    is_uniform = False
                    break

            if not is_uniform or uniform_signature is None:
                continue

            signature_map = signature_maps[attr_idx]
            signature_code = signature_map.get(uniform_signature)
            if signature_code is None:
                signature_code = len(signature_map) + 1
                signature_map[uniform_signature] = signature_code
            object_attr_codes[obj_idx, attr_idx] = signature_code

    return object_attr_codes


def _compute_decision_signature_ids(
    coarse_signatures: Sequence[object],
    fine_signatures: Sequence[object],
) -> np.ndarray:
    decision_ids = np.empty(len(coarse_signatures), dtype=np.int32)
    signature_to_id: Dict[Tuple[object, object], int] = {}

    for obj_idx, signature in enumerate(zip(coarse_signatures, fine_signatures)):
        signature_id = signature_to_id.get(signature)
        if signature_id is None:
            signature_id = len(signature_to_id)
            signature_to_id[signature] = signature_id
        decision_ids[obj_idx] = signature_id

    return decision_ids


def _mask_to_attribute_indices(mask: int) -> List[int]:
    indices: List[int] = []
    current = mask
    while current:
        lowest = current & -current
        indices.append(lowest.bit_length() - 1)
        current ^= lowest
    return indices


def _attribute_mask_to_names(
    mask: int, condition_attributes: Sequence[str]
) -> FrozenSet[str]:
    return frozenset(
        condition_attributes[attr_idx] for attr_idx in _mask_to_attribute_indices(mask)
    )


def _find_conflict_pair_for_selected(
    object_attr_codes: np.ndarray,
    decision_ids: np.ndarray,
    selected_indices: Sequence[int],
) -> Optional[Tuple[int, int]]:
    """
    在当前属性子集下寻找一个仍无法区分、且决策签名不同的对象对。

    当前属性子集下，两对象仍不可区分当且仅当：
    - 在所有已选属性上，它们都具有相同的正编码；
    - 即不存在任一已选属性使该对象对可辨识。
    """
    object_count = object_attr_codes.shape[0]
    grouped: Dict[Tuple[int, ...], Tuple[int, int]] = {}

    if not selected_indices:
        root_key: Tuple[int, ...] = ()
        for obj_idx in range(object_count):
            current_decision = int(decision_ids[obj_idx])
            entry = grouped.get(root_key)
            if entry is None:
                grouped[root_key] = (current_decision, obj_idx)
            elif entry[0] != current_decision:
                return entry[1], obj_idx
        return None

    selected = np.asarray(selected_indices, dtype=np.int32)
    for obj_idx in range(object_count):
        codes = object_attr_codes[obj_idx, selected]
        if np.any(codes <= 0):
            continue

        key = tuple(int(code) for code in codes.tolist())
        current_decision = int(decision_ids[obj_idx])
        entry = grouped.get(key)
        if entry is None:
            grouped[key] = (current_decision, obj_idx)
        elif entry[0] != current_decision:
            return entry[1], obj_idx

    return None


def _pair_clause_mask(
    object_attr_codes: np.ndarray,
    i: int,
    j: int,
    attr_bit_lookup: Sequence[int],
) -> int:
    row_i = object_attr_codes[i]
    row_j = object_attr_codes[j]
    distinguishable = (row_i <= 0) | (row_j <= 0) | (row_i != row_j)
    clause_mask = 0
    for attr_idx in np.flatnonzero(distinguishable):
        clause_mask |= attr_bit_lookup[int(attr_idx)]
    return clause_mask


def _compute_single_attribute_conflict_counts(
    object_attr_codes: np.ndarray,
    decision_ids: np.ndarray,
) -> np.ndarray:
    """
    统计每个属性单独使用时仍无法区分的跨决策对象对数量。
    数值越小，说明该属性越强。
    """
    object_count, attribute_count = object_attr_codes.shape
    if object_count == 0 or attribute_count == 0:
        return np.zeros(attribute_count, dtype=np.int64)

    counts = np.zeros(attribute_count, dtype=np.int64)

    for attr_idx in range(attribute_count):
        code_to_decision_counts: Dict[int, Dict[int, int]] = {}
        column = object_attr_codes[:, attr_idx]

        for obj_idx, code in enumerate(column):
            code_value = int(code)
            if code_value <= 0:
                continue

            decision_value = int(decision_ids[obj_idx])
            decision_counts = code_to_decision_counts.setdefault(code_value, {})
            decision_counts[decision_value] = decision_counts.get(decision_value, 0) + 1

        attr_conflicts = 0
        for decision_counts in code_to_decision_counts.values():
            running = 0
            for count in decision_counts.values():
                attr_conflicts += running * count
                running += count
        counts[attr_idx] = attr_conflicts

    return counts


def _greedy_single_reduct_mask(
    object_attr_codes: np.ndarray,
    decision_ids: np.ndarray,
    attr_conflict_counts: np.ndarray,
    attr_bit_lookup: Sequence[int],
    deadline: Optional[float],
) -> int:
    selected_indices: List[int] = []
    selected_mask = 0

    while True:
        if _deadline_exceeded(deadline):
            raise TimeoutError(_HITTING_TIMEOUT_MESSAGE)

        conflict_pair = _find_conflict_pair_for_selected(
            object_attr_codes,
            decision_ids,
            selected_indices,
        )
        if conflict_pair is None:
            return selected_mask

        clause_mask = _pair_clause_mask(
            object_attr_codes,
            conflict_pair[0],
            conflict_pair[1],
            attr_bit_lookup,
        )
        available_mask = clause_mask & ~selected_mask
        if available_mask == 0:
            raise RuntimeError("当前贪心搜索未找到可用于区分冲突对象对的属性。")

        candidate_indices = _mask_to_attribute_indices(available_mask)
        best_attr = min(
            candidate_indices,
            key=lambda attr_idx: (int(attr_conflict_counts[attr_idx]), attr_idx),
        )
        selected_indices.append(best_attr)
        selected_mask |= attr_bit_lookup[best_attr]


def _exact_single_reduct_mask(
    object_attr_codes: np.ndarray,
    decision_ids: np.ndarray,
    condition_attributes: Sequence[str],
    *,
    timeout: Optional[float] = None,
) -> int:
    attribute_count = len(condition_attributes)
    attr_bit_lookup = tuple(1 << attr_idx for attr_idx in range(attribute_count))
    deadline = time.perf_counter() + timeout if timeout is not None else None
    attr_conflict_counts = _compute_single_attribute_conflict_counts(
        object_attr_codes,
        decision_ids,
    )

    greedy_mask = _greedy_single_reduct_mask(
        object_attr_codes,
        decision_ids,
        attr_conflict_counts,
        attr_bit_lookup,
        deadline,
    )
    best_mask = greedy_mask
    best_size = greedy_mask.bit_count()

    if best_size <= 1:
        return best_mask

    visited_masks: Set[int] = set()

    def search(selected_mask: int, selected_indices: List[int]) -> None:
        nonlocal best_mask
        nonlocal best_size

        if _deadline_exceeded(deadline):
            raise TimeoutError(_HITTING_TIMEOUT_MESSAGE)

        if selected_mask in visited_masks:
            return
        visited_masks.add(selected_mask)

        current_size = len(selected_indices)
        if current_size >= best_size:
            return

        conflict_pair = _find_conflict_pair_for_selected(
            object_attr_codes,
            decision_ids,
            selected_indices,
        )
        if conflict_pair is None:
            best_mask = selected_mask
            best_size = current_size
            return

        if current_size + 1 >= best_size:
            return

        clause_mask = _pair_clause_mask(
            object_attr_codes,
            conflict_pair[0],
            conflict_pair[1],
            attr_bit_lookup,
        )
        available_mask = clause_mask & ~selected_mask
        if available_mask == 0:
            return

        candidate_indices = sorted(
            _mask_to_attribute_indices(available_mask),
            key=lambda attr_idx: (int(attr_conflict_counts[attr_idx]), attr_idx),
        )
        for attr_idx in candidate_indices:
            selected_indices.append(attr_idx)
            search(selected_mask | attr_bit_lookup[attr_idx], selected_indices)
            selected_indices.pop()

    search(0, [])
    return best_mask


def _compute_delta_sets(
    tolerance_classes: Sequence[Set[int]], decisions: Sequence[str]
) -> List[Set[str]]:
    delta_sets: List[Set[str]] = []
    for tol_class in tolerance_classes:
        delta_sets.append({decisions[idx] for idx in tol_class})
    return delta_sets


def _compute_tau_sets(
    blocks_zero_based: Sequence[FrozenSet[int]], decisions: Sequence[str]
) -> List[Set[str]]:
    tau_sets: List[Set[str]] = []
    for block in blocks_zero_based:
        tau_sets.append({decisions[idx] for idx in block})
    return tau_sets


def _is_positive_label_value(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "y"}
    return bool(value)


def _compute_label_masks(
    label_values: Sequence[Sequence[object]],
) -> np.ndarray:
    label_masks = np.zeros(len(label_values), dtype=np.uint64)
    for row_idx, row in enumerate(label_values):
        mask = 0
        for bit_idx, value in enumerate(row):
            if _is_positive_label_value(value):
                mask |= 1 << bit_idx
        label_masks[row_idx] = mask
    return label_masks


def _attribute_clause_for_pair(
    condition_attributes: Sequence[str],
    phi_block_indices_by_object: Sequence[Sequence[int]],
    block_missing_all: Sequence[Sequence[bool]],
    block_observed_values: Sequence[Sequence[FrozenSet[str]]],
    i: int,
    j: int,
) -> FrozenSet[str]:
    """
    优化版本：减少重复的集合操作，提前退出循环。
    """
    phi_i_indices = phi_block_indices_by_object[i]
    phi_j_indices = phi_block_indices_by_object[j]

    attribute_clause: Set[str] = set()
    
    # 预先缓存块信息，避免重复访问
    phi_i_blocks = [(idx, block_missing_all[idx], block_observed_values[idx]) 
                     for idx in phi_i_indices]
    phi_j_blocks = [(idx, block_missing_all[idx], block_observed_values[idx]) 
                     for idx in phi_j_indices]
    
    for attr_index, attr_name in enumerate(condition_attributes):
        found = False
        # 快速路径：如果对象i的所有块在该属性上都缺失，或对象j的所有块都缺失
        i_all_missing = all(missing_all[attr_index] for _, missing_all, _ in phi_i_blocks)
        j_all_missing = all(missing_all[attr_index] for _, missing_all, _ in phi_j_blocks)
        if i_all_missing or j_all_missing:
            attribute_clause.add(attr_name)
            continue
        
        # 检查是否存在块对使得该属性可辨识
        for block_i_idx, missing_all_i, observed_values_i in phi_i_blocks:
            missing_i = missing_all_i[attr_index]
            observed_i = observed_values_i[attr_index]
            
            for block_j_idx, missing_all_j, observed_values_j in phi_j_blocks:
                missing_j = missing_all_j[attr_index]
                observed_j = observed_values_j[attr_index]
                
                # 如果任一块的该属性全部缺失，则需要该属性
                if missing_i or missing_j:
                    attribute_clause.add(attr_name)
                    found = True
                    break

                # 如果两个块的观察值集合不同，则需要该属性
                if observed_i and observed_j and observed_i != observed_j:
                    attribute_clause.add(attr_name)
                    found = True
                    break
            if found:
                break

    if not attribute_clause:
        raise ValueError(
            f"无法为对象对 ({i + 1}, {j + 1}) 找到可辨识属性。"
        )

    return frozenset(attribute_clause)


def _compute_discernibility_clauses_serial(
    condition_attributes: Sequence[str],
    phi_block_indices_by_object: Sequence[Sequence[int]],
    coarse_sets: Sequence[FrozenSet[str]],
    fine_sets: Sequence[FrozenSet[str]],
    block_missing_all: Sequence[Sequence[bool]],
    block_observed_values: Sequence[Sequence[FrozenSet[str]]],
) -> List[FrozenSet[str]]:
    """
    优化版本：使用集合快速比较，减少不必要的属性子句计算。
    """
    object_count = len(phi_block_indices_by_object)
    clauses: List[FrozenSet[str]] = []
    # 使用集合存储已计算的子句，避免重复（虽然理论上不应该重复）
    seen_clauses: Set[FrozenSet[str]] = set()
    
    for i in range(object_count):
        coarse_i = coarse_sets[i]
        fine_i = fine_sets[i]
        for j in range(i + 1, object_count):
            # 快速路径：如果coarse和fine集合都相同，跳过
            coarse_j = coarse_sets[j]
            fine_j = fine_sets[j]
            if coarse_i == coarse_j and fine_i == fine_j:
                continue
            
            clause = _attribute_clause_for_pair(
                condition_attributes,
                phi_block_indices_by_object,
                block_missing_all,
                block_observed_values,
                i,
                j,
            )
            # 避免重复子句（虽然理论上不应该发生）
            if clause not in seen_clauses:
                clauses.append(clause)
                seen_clauses.add(clause)
    return clauses


def _discernibility_worker_initializer(
    condition_attributes: Sequence[str],
    phi_block_indices_by_object: Sequence[Sequence[int]],
    coarse_sets: Sequence[FrozenSet[str]],
    fine_sets: Sequence[FrozenSet[str]],
    block_missing_all: Sequence[Sequence[bool]],
    block_observed_values: Sequence[Sequence[FrozenSet[str]]],
) -> None:
    global _DISCERN_DATA
    global _DISCERN_PHI_BY_OBJECT
    global _DISCERN_CONDITION_ATTRIBUTES
    global _DISCERN_COARSE_SETS
    global _DISCERN_FINE_SETS
    global _DISCERN_PHI_INDICES
    global _DISCERN_BLOCK_MISSING_ALL
    global _DISCERN_BLOCK_OBSERVED

    _DISCERN_DATA = None
    _DISCERN_PHI_BY_OBJECT = None
    _DISCERN_CONDITION_ATTRIBUTES = condition_attributes
    _DISCERN_COARSE_SETS = coarse_sets
    _DISCERN_FINE_SETS = fine_sets
    _DISCERN_PHI_INDICES = phi_block_indices_by_object
    _DISCERN_BLOCK_MISSING_ALL = block_missing_all
    _DISCERN_BLOCK_OBSERVED = block_observed_values


def _discernibility_worker_task(i: int) -> List[FrozenSet[str]]:
    if (
        _DISCERN_CONDITION_ATTRIBUTES is None
        or _DISCERN_COARSE_SETS is None
        or _DISCERN_FINE_SETS is None
        or _DISCERN_PHI_INDICES is None
        or _DISCERN_BLOCK_MISSING_ALL is None
        or _DISCERN_BLOCK_OBSERVED is None
    ):
        raise RuntimeError("Discernibility worker not initialized.")

    condition_attributes = _DISCERN_CONDITION_ATTRIBUTES
    coarse_sets = _DISCERN_COARSE_SETS
    fine_sets = _DISCERN_FINE_SETS
    phi_indices = _DISCERN_PHI_INDICES
    block_missing_all = _DISCERN_BLOCK_MISSING_ALL
    block_observed_values = _DISCERN_BLOCK_OBSERVED

    object_count = len(phi_indices)
    if i >= object_count:
        return []

    clauses_for_i: List[FrozenSet[str]] = []
    coarse_i = coarse_sets[i]
    fine_i = fine_sets[i]

    for j in range(i + 1, object_count):
        if coarse_i != coarse_sets[j] or fine_i != fine_sets[j]:
            clause = _attribute_clause_for_pair(
                condition_attributes,
                phi_indices,
                block_missing_all,
                block_observed_values,
                i,
                j,
            )
            clauses_for_i.append(clause)

    return clauses_for_i


def _compute_discernibility_clauses_parallel(
    condition_attributes: Sequence[str],
    phi_block_indices_by_object: Sequence[Sequence[int]],
    coarse_sets: Sequence[FrozenSet[str]],
    fine_sets: Sequence[FrozenSet[str]],
    block_missing_all: Sequence[Sequence[bool]],
    block_observed_values: Sequence[Sequence[FrozenSet[str]]],
    max_workers: int,
) -> List[FrozenSet[str]]:
    object_count = len(phi_block_indices_by_object)
    if object_count <= 1:
        return []

    phi_tuple = tuple(tuple(indices) for indices in phi_block_indices_by_object)
    coarse_tuple = tuple(coarse_sets)
    fine_tuple = tuple(fine_sets)
    condition_attributes_tuple = tuple(condition_attributes)
    block_missing_tuple = tuple(tuple(row) for row in block_missing_all)
    block_observed_tuple = tuple(tuple(values) for values in block_observed_values)

    task_count = max(1, object_count - 1)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_discernibility_worker_initializer,
        initargs=(
            condition_attributes_tuple,
            phi_tuple,
            coarse_tuple,
            fine_tuple,
            block_missing_tuple,
            block_observed_tuple,
        ),
    ) as executor:
        chunk_size = _resolve_map_chunk_size(task_count, max_workers)
        tasks = range(object_count - 1)
        clauses: List[FrozenSet[str]] = []
        for clause_list in executor.map(
            _discernibility_worker_task, tasks, chunksize=chunk_size
        ):
            if clause_list:
                clauses.extend(clause_list)
    return clauses


def _attribute_discernible_between_blocks(
    data: Sequence[Sequence[str]],
    attr_index: int,
    block_a: Iterable[int],
    block_b: Iterable[int],
) -> bool:
    values_a = [data[idx][attr_index] for idx in block_a]
    values_b = [data[idx][attr_index] for idx in block_b]

    if not values_a or not values_b:
        return False

    if all(value == MISSING_VALUE for value in values_a) or all(
        value == MISSING_VALUE for value in values_b
    ):
        return True

    observed_a = {value for value in values_a if value != MISSING_VALUE}
    observed_b = {value for value in values_b if value != MISSING_VALUE}
    for value_a in observed_a:
        for value_b in observed_b:
            if value_a != value_b:
                return True
    return False


def _discernibility_attributes(
    row_x: Sequence[str], row_y: Sequence[str], attribute_names: Sequence[str]
) -> Set[str]:
    attributes: Set[str] = set()
    for name, value_x, value_y in zip(attribute_names, row_x, row_y):
        if value_x == MISSING_VALUE or value_y == MISSING_VALUE:
            continue
        if value_x != value_y:
            attributes.add(name)
    return attributes


def _build_discernibility_clauses(
    table: IncompleteDecisionTable,
    delta_sets: Sequence[Set[str]],
    blocks_zero_based: Sequence[FrozenSet[int]],
    tau_sets: Sequence[Set[str]],
) -> List[FrozenSet[str]]:
    data = table.condition_values
    clauses: Set[FrozenSet[str]] = set()

    for x_idx, delta_x in enumerate(delta_sets):
        row_x = data[x_idx]
        for block_idx, block in enumerate(blocks_zero_based):
            if tau_sets[block_idx].issubset(delta_x):
                continue

            relevant_objects = [
                obj_idx
                for obj_idx in block
                if not delta_sets[obj_idx].issubset(delta_x)
            ]
            if not relevant_objects:
                continue

            for obj_idx in relevant_objects:
                attributes = _discernibility_attributes(
                    row_x,
                    data[obj_idx],
                    table.condition_attributes,
                )
                if not attributes:
                    continue
                clauses.add(frozenset(attributes))

    return sorted(clauses, key=lambda clause: (len(clause), sorted(clause)))


def _prune_clauses(clauses: Sequence[FrozenSet[str]]) -> List[FrozenSet[str]]:
    ordered = sorted(clauses, key=lambda clause: (len(clause), sorted(clause)))
    pruned: List[FrozenSet[str]] = []
    for clause in ordered:
        if any(existing.issubset(clause) for existing in pruned):
            continue
        pruned = [existing for existing in pruned if not clause.issubset(existing)]
        pruned.append(clause)
    return pruned


_HITTING_TIMEOUT_MESSAGE = "最小击中集搜索超时。"


def _deadline_exceeded(deadline: Optional[float]) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _add_hitting_result(
    candidate: Set[str], results: Set[FrozenSet[str]]
) -> bool:
    candidate_frozen = frozenset(sorted(candidate))
    if any(existing.issubset(candidate_frozen) for existing in results):
        return False
    supersets = {existing for existing in results if candidate_frozen.issubset(existing)}
    if supersets:
        results.difference_update(supersets)
    results.add(candidate_frozen)
    return True


def _search_minimal_hitting_sets(
    selected: Set[str],
    unsatisfied: List[Set[str]],
    results: Set[FrozenSet[str]],
    limit: Optional[int],
    deadline: Optional[float],
) -> bool:
    if limit is not None and limit <= 0:
        return True

    if any(existing.issubset(selected) for existing in results):
        return False

    if _deadline_exceeded(deadline):
        raise TimeoutError(_HITTING_TIMEOUT_MESSAGE)

    if not unsatisfied:
        added = _add_hitting_result(selected, results)
        if limit is not None and added and len(results) >= limit:
            return True
        return False

    clause = min(unsatisfied, key=len)
    for attribute in sorted(clause):
        if _deadline_exceeded(deadline):
            raise TimeoutError(_HITTING_TIMEOUT_MESSAGE)
        new_selected = set(selected)
        new_selected.add(attribute)
        new_unsatisfied = [
            other for other in unsatisfied if attribute not in other
        ]
        should_stop = _search_minimal_hitting_sets(
            new_selected,
            new_unsatisfied,
            results,
            limit,
            deadline,
        )
        if should_stop:
            return True
    return False


def _build_hitting_seed_tasks(
    clause_sets: Sequence[FrozenSet[str]],
    split_depth: int,
    max_tasks: int,
) -> List[HittingSeed]:
    initial_unsatisfied = [set(clause) for clause in clause_sets]
    stack: List[Tuple[Set[str], List[Set[str]], int]] = [
        (set(), initial_unsatisfied, split_depth)
    ]
    seeds: List[HittingSeed] = []

    while stack:
        selected, unsatisfied, depth = stack.pop()
        if depth <= 0 or not unsatisfied or len(seeds) >= max_tasks:
            seeds.append(
                (
                    frozenset(sorted(selected)),
                    tuple(frozenset(clause) for clause in unsatisfied),
                )
            )
            continue

        clause = min(unsatisfied, key=len)
        for attribute in sorted(clause):
            new_selected = set(selected)
            new_selected.add(attribute)
            new_unsatisfied = [
                set(other) for other in unsatisfied if attribute not in other
            ]
            stack.append((new_selected, new_unsatisfied, depth - 1))

    return seeds


def _minimal_hitting_sets_worker(seed: HittingSeed) -> List[FrozenSet[str]]:
    selected_seed, unsatisfied_seed = seed
    selected = set(selected_seed)
    unsatisfied = [set(clause) for clause in unsatisfied_seed]
    results: Set[FrozenSet[str]] = set()
    _search_minimal_hitting_sets(selected, unsatisfied, results, limit=None, deadline=None)
    return sorted(results, key=lambda reduct: (len(reduct), sorted(reduct)))


def _minimal_hitting_sets_serial(
    clause_sets: Sequence[FrozenSet[str]],
    limit: Optional[int] = None,
    deadline: Optional[float] = None,
) -> List[FrozenSet[str]]:
    remaining = [set(clause) for clause in clause_sets]
    if not remaining:
        return [frozenset()]

    results: Set[FrozenSet[str]] = set()
    _search_minimal_hitting_sets(set(), remaining, results, limit, deadline)
    return sorted(results, key=lambda reduct: (len(reduct), sorted(reduct)))


def _minimal_hitting_sets_parallel(
    clause_sets: Sequence[FrozenSet[str]],
    deadline: Optional[float] = None,
) -> List[FrozenSet[str]]:
    if deadline is not None:
        # 无法安全地在多进程中共享统一超时时间；退化为串行求解。
        return _minimal_hitting_sets_serial(clause_sets, limit=None, deadline=deadline)

    worker_budget = _resolve_parallel_worker_budget()
    if worker_budget <= 1:
        return _minimal_hitting_sets_serial(clause_sets)

    split_depth_env = os.getenv("ALGORITHM1_HITTING_SPLIT_DEPTH")
    try:
        split_depth = (
            int(split_depth_env) if split_depth_env is not None else 2
        )
    except ValueError:
        split_depth = 2
    split_depth = max(1, split_depth)

    max_tasks_env = os.getenv("ALGORITHM1_HITTING_MAX_TASKS")
    try:
        max_tasks = int(max_tasks_env) if max_tasks_env is not None else 0
    except ValueError:
        max_tasks = 0
    if max_tasks <= 0:
        max_tasks = worker_budget * 16

    seeds = _build_hitting_seed_tasks(clause_sets, split_depth, max_tasks)
    if not seeds:
        return _minimal_hitting_sets_serial(clause_sets)

    max_workers = min(worker_budget, len(seeds))
    if max_workers <= 1:
        return _minimal_hitting_sets_serial(clause_sets)

    chunk_size = _resolve_map_chunk_size(len(seeds), max_workers)
    merged_results: Set[FrozenSet[str]] = set()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for partial_results in executor.map(
            _minimal_hitting_sets_worker, seeds, chunksize=chunk_size
        ):
            for candidate in partial_results:
                _add_hitting_result(set(candidate), merged_results)

    if not merged_results:
        return _minimal_hitting_sets_serial(clause_sets)

    return sorted(merged_results, key=lambda reduct: (len(reduct), sorted(reduct)))


def _minimal_hitting_sets(
    clauses: Sequence[FrozenSet[str]],
    *,
    limit: Optional[int] = None,
    deadline: Optional[float] = None,
) -> List[FrozenSet[str]]:
    clause_sets = tuple(frozenset(clause) for clause in clauses if clause)
    if not clause_sets:
        return [frozenset()]

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 必须为正整数。")
        # 限定结果数量时，串行搜索便于及早剪枝与判断。
        return _minimal_hitting_sets_serial(
            clause_sets,
            limit=limit,
            deadline=deadline,
        )

    if deadline is not None:
        return _minimal_hitting_sets_serial(
            clause_sets,
            limit=None,
            deadline=deadline,
        )

    literal_count = sum(len(clause) for clause in clause_sets)
    if _should_parallelize_hitting_sets(len(clause_sets), literal_count):
        try:
            return _minimal_hitting_sets_parallel(clause_sets, deadline=None)
        except Exception:
            pass

    return _minimal_hitting_sets_serial(clause_sets)


def _is_hitting_set(candidate: Set[str], clauses: Sequence[Set[str]]) -> bool:
    return all(candidate.intersection(clause) for clause in clauses)


def _greedy_minimal_hitting_set(
    clauses: Sequence[FrozenSet[str]],
) -> FrozenSet[str]:
    clause_sets = [set(clause) for clause in clauses if clause]
    if not clause_sets:
        return frozenset()

    uncovered = list(clause_sets)
    selected: Set[str] = set()

    while uncovered:
        frequency: Dict[str, int] = {}
        for clause in uncovered:
            for attribute in clause:
                frequency[attribute] = frequency.get(attribute, 0) + 1
        attribute, _ = min(
            frequency.items(), key=lambda item: (-item[1], item[0])
        )
        selected.add(attribute)
        uncovered = [clause for clause in uncovered if attribute not in clause]

    for attribute in sorted(list(selected)):
        candidate = set(selected)
        candidate.remove(attribute)
        if _is_hitting_set(candidate, clause_sets):
            selected = candidate

    return frozenset(sorted(selected))


def _greedy_hitting_sets_varied(
    clauses: Sequence[FrozenSet[str]],
    max_count: int = 5,
) -> List[FrozenSet[str]]:
    """
    生成多个不同大小的贪心约简候选，用于提供更多选择。
    通过改变贪心选择的策略（如随机化、不同的频率计算方式）生成不同的约简。
    """
    clause_sets = [set(clause) for clause in clauses if clause]
    if not clause_sets:
        return [frozenset()]
    
    results: List[FrozenSet[str]] = []
    all_attributes = set()
    for clause in clause_sets:
        all_attributes.update(clause)
    
    # 生成最小约简
    minimal = _greedy_minimal_hitting_set(clauses)
    if minimal:
        results.append(minimal)
    
    # 如果最小约简太小，生成一些包含更多属性的约简
    if len(minimal) <= 2 and len(all_attributes) > len(minimal):
        # 策略1: 在最小约简基础上，添加一些高频属性
        if minimal:
            extended1 = set(minimal)
            frequency: Dict[str, int] = {}
            for clause in clause_sets:
                for attr in clause:
                    if attr not in extended1:
                        frequency[attr] = frequency.get(attr, 0) + 1
            # 添加最多3个高频属性
            sorted_attrs = sorted(frequency.items(), key=lambda x: (-x[1], x[0]))[:3]
            for attr, _ in sorted_attrs:
                extended1.add(attr)
                if len(extended1) >= min(10, len(all_attributes)):
                    break
            if len(extended1) > len(minimal):
                results.append(frozenset(sorted(extended1)))
        
        # 策略2: 使用不同的贪心策略（优先选择覆盖更多子句的属性）
        uncovered = list(clause_sets)
        selected2: Set[str] = set()
        while uncovered and len(selected2) < min(15, len(all_attributes)):
            # 计算每个属性能覆盖多少未覆盖的子句
            coverage_count: Dict[str, int] = {}
            for attr in all_attributes:
                if attr not in selected2:
                    count = sum(1 for clause in uncovered if attr in clause)
                    coverage_count[attr] = count
            if not coverage_count:
                break
            # 选择覆盖最多子句的属性
            best_attr = max(coverage_count.items(), key=lambda x: (x[1], x[0]))[0]
            selected2.add(best_attr)
            uncovered = [clause for clause in uncovered if best_attr not in clause]
        
        if selected2 and frozenset(sorted(selected2)) not in results:
            results.append(frozenset(sorted(selected2)))
    
    # 去重并限制数量
    unique_results = []
    seen = set()
    for r in results:
        if r not in seen:
            unique_results.append(r)
            seen.add(r)
            if len(unique_results) >= max_count:
                break
    
    return sorted(unique_results, key=lambda r: (len(r), sorted(r)))


def algorithm1_maximal_compatibility_blocks(
    data: Sequence[Sequence[str]],
) -> List[FrozenSet[int]]:
    """
    按“算法 1” 的思路计算不完备决策信息系统的全体极大相容块 φ(A)。
    返回的每个集合使用 1 开始的对象编号，方便与手稿中的描述对照。
    """
    tolerance_matrix = compute_tolerance_matrix(data)
    tolerance_classes = compute_tolerance_classes(tolerance_matrix)

    # 直接基于容差类交集生成候选块，避免构造 n×n 的集合交矩阵。
    candidates_zero_based: List[FrozenSet[int]] = []
    seen_blocks: Set[FrozenSet[int]] = set()
    for i, s_i in enumerate(tolerance_classes):
        for j in s_i:
            if j < i:
                continue
            block = frozenset(s_i & tolerance_classes[j])
            if not block or block in seen_blocks:
                continue
            seen_blocks.add(block)
            if _is_compatibility_block(block, tolerance_matrix):
                candidates_zero_based.append(block)

    unique_candidates = _unique_frozensets(candidates_zero_based)
    maximal_blocks_zero_based = _maximal_sets(unique_candidates)

    # 转换为 1 基下标，方便对手稿结果。
    return [
        frozenset({idx + 1 for idx in block}) for block in maximal_blocks_zero_based
    ]


def algorithm2_single_label_reducts(
    table: IncompleteDecisionTable,
) -> List[FrozenSet[str]]:
    """
    复现手稿中的算法 2，计算不完备单标记决策信息系统的全部属性约简。
    返回的每个约简使用条件属性名表示。
    """

    tolerance_matrix = compute_tolerance_matrix(table.condition_values)
    tolerance_classes = compute_tolerance_classes(tolerance_matrix)
    delta_sets = _compute_delta_sets(tolerance_classes, table.decision_values)

    blocks_one_based = algorithm1_maximal_compatibility_blocks(
        table.condition_values
    )
    blocks_zero_based = _to_zero_based_blocks(blocks_one_based)
    tau_sets = _compute_tau_sets(blocks_zero_based, table.decision_values)

    clauses = _build_discernibility_clauses(
        table, delta_sets, blocks_zero_based, tau_sets
    )
    pruned_clauses = _prune_clauses(clauses)
    reducts = _minimal_hitting_sets(pruned_clauses)
    return reducts


def algorithm3_multi_label_reducts(
    table: IncompleteMultiLabelDecisionTable,
    *,
    max_reducts: Optional[int] = None,
    timeout: Optional[float] = None,
    prefer_greedy: bool = False,
) -> List[FrozenSet[str]]:
    """
    复现手稿中的算法 3，计算不完备多标记决策信息系统的全部互补决策约简。
    返回的每个约简使用条件属性名表示。
    """

    if max_reducts is not None and max_reducts <= 0:
        raise ValueError("max_reducts 必须为正整数。")

    if timeout is not None and timeout <= 0:
        raise ValueError("timeout 必须为正数秒。")

    limit = max_reducts
    if prefer_greedy and limit is None:
        limit = 1

    data = table.condition_values
    blocks_one_based = algorithm1_maximal_compatibility_blocks(data)
    blocks_zero_based = _to_zero_based_blocks(blocks_one_based)
    object_count = table.object_count
    phi_by_object = _compute_phi_by_object(blocks_zero_based, object_count)
    condition_attributes = tuple(table.condition_attributes)
    attribute_count = len(condition_attributes)
    block_missing_all, block_observed_values = _build_block_attribute_cache(
        data,
        blocks_zero_based,
        attribute_count,
    )
    block_to_index = {block: idx for idx, block in enumerate(blocks_zero_based)}
    phi_block_indices_by_object = [
        [block_to_index[block] for block in phi_blocks]
        for phi_blocks in phi_by_object
    ]
    object_attr_codes = _build_object_attribute_codes(
        phi_block_indices_by_object,
        block_missing_all,
        block_observed_values,
        attribute_count,
    )

    label_masks = _compute_label_masks(table.label_values)
    block_union_masks: List[int] = []
    block_intersection_masks: List[int] = []
    all_label_bits = (1 << len(table.label_names)) - 1

    for block in blocks_zero_based:
        union_mask = 0
        intersection_mask = all_label_bits
        has_object = False
        for obj_idx in block:
            has_object = True
            obj_mask = int(label_masks[obj_idx])
            union_mask |= obj_mask
            intersection_mask &= obj_mask
        if not has_object:
            intersection_mask = 0
        block_union_masks.append(union_mask)
        block_intersection_masks.append(intersection_mask)

    coarse_masks: List[int] = []
    fine_masks: List[int] = []
    for block_indices in phi_block_indices_by_object:
        coarse_mask = 0
        fine_mask: Optional[int] = None
        for block_idx in block_indices:
            coarse_mask |= block_union_masks[block_idx]
            block_common_mask = block_intersection_masks[block_idx]
            if fine_mask is None:
                fine_mask = block_common_mask
            else:
                fine_mask &= block_common_mask
        coarse_masks.append(coarse_mask)
        fine_masks.append(0 if fine_mask is None else fine_mask)

    decision_signature_ids = _compute_decision_signature_ids(coarse_masks, fine_masks)

    if limit == 1 and not prefer_greedy:
        exact_mask = _exact_single_reduct_mask(
            object_attr_codes,
            decision_signature_ids,
            condition_attributes,
            timeout=timeout,
        )
        return [_attribute_mask_to_names(exact_mask, condition_attributes)]

    if limit == 1 and prefer_greedy:
        attr_bit_lookup = tuple(1 << attr_idx for attr_idx in range(attribute_count))
        attr_conflict_counts = _compute_single_attribute_conflict_counts(
            object_attr_codes,
            decision_signature_ids,
        )
        greedy_mask = _greedy_single_reduct_mask(
            object_attr_codes,
            decision_signature_ids,
            attr_conflict_counts,
            attr_bit_lookup,
            time.perf_counter() + timeout if timeout is not None else None,
        )
        return [_attribute_mask_to_names(greedy_mask, condition_attributes)]

    if _should_parallelize_discernibility(object_count):
        max_workers = _resolve_parallel_worker_count(object_count)
        try:
            clauses = _compute_discernibility_clauses_parallel(
                condition_attributes,
                phi_block_indices_by_object,
                coarse_masks,
                fine_masks,
                block_missing_all,
                block_observed_values,
                max_workers,
            )
        except Exception:
            clauses = _compute_discernibility_clauses_serial(
                condition_attributes,
                phi_block_indices_by_object,
                coarse_masks,
                fine_masks,
                block_missing_all,
                block_observed_values,
            )
    else:
        clauses = _compute_discernibility_clauses_serial(
            condition_attributes,
            phi_block_indices_by_object,
            coarse_masks,
            fine_masks,
            block_missing_all,
            block_observed_values,
        )

    if not clauses:
        return [frozenset()]

    unique_clauses = sorted(
        {clause for clause in clauses}, key=lambda clause: (len(clause), sorted(clause))
    )
    pruned_clauses = _prune_clauses(unique_clauses)

    if prefer_greedy:
        if limit == 1:
            greedy_reduct = _greedy_minimal_hitting_set(pruned_clauses)
            return [greedy_reduct]
        else:
            # 生成多个不同大小的贪心约简
            greedy_reducts = _greedy_hitting_sets_varied(pruned_clauses, max_count=limit or 5)
            return greedy_reducts

    deadline = time.perf_counter() + timeout if timeout is not None else None

    try:
        reducts = _minimal_hitting_sets(
            pruned_clauses,
            limit=limit,
            deadline=deadline,
        )
    except TimeoutError:
        if prefer_greedy:
            greedy_reduct = _greedy_minimal_hitting_set(pruned_clauses)
            return [greedy_reduct]
        raise

    return reducts


def _example_decision_table() -> List[List[str]]:
    """
    文中“表 1 决策信息系统 1”的条件属性数据。
    不包含决策属性列，算法 1 仅依赖条件属性。
    """
    return [
        ["A1", "B2", "*", "*", "*", "F2", "*", "*", "*", "*"],
        ["A1", "B2", "*", "*", "*", "F1", "*", "H2", "I2", "J1"],
        ["A1", "B1", "*", "*", "*", "F1", "*", "H2", "I2", "J1"],
        ["*", "*", "C1", "D1", "*", "F2", "*", "H1", "I2", "J1"],
        ["*", "*", "*", "*", "E1", "F2", "*", "*", "*", "*"],
        ["*", "*", "*", "*", "E2", "F1", "*", "*", "*", "*"],
        ["A2", "B1", "C1", "D1", "E2", "F2", "G1", "H1", "I2", "J1"],
        ["*", "*", "*", "*", "*", "*", "G1", "H2", "*", "*"],
        ["*", "*", "*", "*", "*", "*", "G2", "H1", "*", "*"],
        ["A1", "B1", "C2", "D1", "E1", "F1", "G2", "H2", "I2", "J1"],
        ["A1", "B2", "*", "*", "*", "F1", "*", "H2", "I2", "J2"],
        ["*", "*", "C1", "D2", "*", "F2", "*", "H1", "I2", "J2"],
    ]


def _example_decision_system() -> IncompleteDecisionTable:
    attributes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    decisions = ["a", "a", "b", "c", "d", "d", "d", "e", "e", "e", "f", "f"]
    return IncompleteDecisionTable(
        condition_attributes=attributes,
        condition_values=_example_decision_table(),
        decision_values=decisions,
    )


def _example_multilabel_condition_values() -> List[List[str]]:
    return [
        ["1", "*", "1", "1", "*"],
        ["*", "2", "1", "*", "2"],
        ["1", "2", "*", "1", "1"],
        ["*", "1", "2", "2", "*"],
        ["2", "*", "2", "*", "1"],
        ["1", "1", "1", "*", "2"],
        ["*", "2", "*", "2", "1"],
        ["2", "1", "*", "1", "1"],
    ]


def _example_multilabel_label_values() -> List[List[int]]:
    return [
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]


def _example_multilabel_decision_system() -> IncompleteMultiLabelDecisionTable:
    attributes = ["a", "b", "c", "d", "e"]
    label_names = ["l1", "l2", "l3"]
    return IncompleteMultiLabelDecisionTable(
        condition_attributes=attributes,
        condition_values=_example_multilabel_condition_values(),
        label_names=label_names,
        label_values=_example_multilabel_label_values(),
    )


def _expected_blocks() -> Set[FrozenSet[int]]:
    """
    手稿中给出的极大相容块集合 φ(A)。
    """
    expected = [
        {2, 6, 8},
        {3, 6, 8},
        {3, 10},
        {1, 4, 5, 9},
        {4, 7},
        {6, 9},
        {1, 5, 8},
        {6, 8, 11},
        {1, 5, 9, 12},
    ]
    return {frozenset(block) for block in expected}


def _verify_example() -> None:
    data = _example_decision_table()
    computed_blocks = set(algorithm1_maximal_compatibility_blocks(data))
    expected_blocks = _expected_blocks()

    if computed_blocks != expected_blocks:
        missing = expected_blocks - computed_blocks
        extra = computed_blocks - expected_blocks
        message = [
            "算法 1 验证失败：",
            f"  缺失的块: {sorted(missing, key=lambda s: (len(s), sorted(s)))}",
            f"  多余的块: {sorted(extra, key=lambda s: (len(s), sorted(s)))}",
        ]
        raise AssertionError("\n".join(message))

    print("算法 1 验证成功，极大相容块集合与手稿算例完全一致。")
    for block in sorted(computed_blocks, key=lambda s: (len(s), sorted(s))):
        print(f"块大小 {len(block)}: {sorted(block)}")


def _expected_algorithm2_reducts() -> Set[FrozenSet[str]]:
    return {frozenset({"B", "E", "F", "G", "H", "J"})}


def _expected_algorithm3_reducts() -> Set[FrozenSet[str]]:
    return {frozenset({"b", "d", "e"})}


def _verify_algorithm2_example() -> None:
    system = _example_decision_system()
    computed_reducts = set(algorithm2_single_label_reducts(system))
    expected_reducts = _expected_algorithm2_reducts()

    if computed_reducts != expected_reducts:
        missing = expected_reducts - computed_reducts
        extra = computed_reducts - expected_reducts
        message = [
            "算法 2 验证失败：",
            f"  缺失的约简: {sorted(missing, key=lambda s: (len(s), sorted(s)))}",
            f"  多余的约简: {sorted(extra, key=lambda s: (len(s), sorted(s)))}",
        ]
        raise AssertionError("\n".join(message))

    print("算法 2 验证成功，约简结果与手稿算例完全一致。")
    for reduct in sorted(computed_reducts, key=lambda s: (len(s), sorted(s))):
        print(f"约简大小 {len(reduct)}: {sorted(reduct)}")


def _verify_algorithm3_example() -> None:
    system = _example_multilabel_decision_system()
    computed_reducts = set(algorithm3_multi_label_reducts(system))
    expected_reducts = _expected_algorithm3_reducts()

    if computed_reducts != expected_reducts:
        missing = expected_reducts - computed_reducts
        extra = computed_reducts - expected_reducts
        message = [
            "算法 3 验证失败：",
            f"  缺失的约简: {sorted(missing, key=lambda s: (len(s), sorted(s)))}",
            f"  多余的约简: {sorted(extra, key=lambda s: (len(s), sorted(s)))}",
        ]
        raise AssertionError("\n".join(message))

    print("算法 3 验证成功，互补决策约简结果与预期算例一致。")
    for reduct in sorted(computed_reducts, key=lambda s: (len(s), sorted(s))):
        print(f"互补决策约简大小 {len(reduct)}: {sorted(reduct)}")


if __name__ == "__main__":
    _verify_example()
    _verify_algorithm2_example()
    _verify_algorithm3_example()
