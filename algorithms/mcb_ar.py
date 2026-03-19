from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


MISSING_VALUE = "*"


@dataclass(frozen=True)
class IncompleteSingleLabelDecisionTable:
    """Incomplete single-label decision table used by the MCB-AR algorithm."""

    condition_attributes: Tuple[str, ...]
    condition_values: Tuple[Tuple[str, ...], ...]
    decision_attribute: str
    decision_values: Tuple[str, ...]

    @property
    def object_count(self) -> int:
        return len(self.condition_values)

    @property
    def attribute_count(self) -> int:
        return len(self.condition_attributes)


@dataclass(frozen=True)
class BlockDecisionTable:
    """Most fully described decision table constructed on maximal consistent blocks."""

    condition_attributes: Tuple[str, ...]
    condition_values: Tuple[Tuple[str, ...], ...]
    decision_attribute: str
    decision_values: Tuple[FrozenSet[str], ...]
    source_blocks: Tuple[FrozenSet[int], ...]


def _normalize_rows(rows: Sequence[Sequence[str]]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(tuple(str(value) for value in row) for row in rows)


def build_incomplete_single_label_table(
    condition_attributes: Sequence[str],
    condition_values: Sequence[Sequence[str]],
    decision_values: Sequence[str],
    *,
    decision_attribute: str = "d",
) -> IncompleteSingleLabelDecisionTable:
    normalized_conditions = _normalize_rows(condition_values)
    if len(normalized_conditions) != len(decision_values):
        raise ValueError("Condition rows and decision values must have the same length.")

    if normalized_conditions and any(
        len(row) != len(condition_attributes) for row in normalized_conditions
    ):
        raise ValueError("Each condition row must match the number of condition attributes.")

    return IncompleteSingleLabelDecisionTable(
        condition_attributes=tuple(condition_attributes),
        condition_values=normalized_conditions,
        decision_attribute=str(decision_attribute),
        decision_values=tuple(str(value) for value in decision_values),
    )


def compute_tolerance_matrix(
    condition_values: Sequence[Sequence[str]],
    *,
    missing_value: str = MISSING_VALUE,
) -> np.ndarray:
    """Compute the tolerance-relation matrix used by MCB-AR."""

    rows = _normalize_rows(condition_values)
    object_count = len(rows)
    matrix = np.zeros((object_count, object_count), dtype=bool)
    for i in range(object_count):
        matrix[i, i] = True
        row_i = rows[i]
        for j in range(i + 1, object_count):
            row_j = rows[j]
            compatible = True
            for value_i, value_j in zip(row_i, row_j):
                if (
                    value_i != missing_value
                    and value_j != missing_value
                    and value_i != value_j
                ):
                    compatible = False
                    break
            matrix[i, j] = compatible
            matrix[j, i] = compatible
    return matrix


def compute_tolerance_classes(tolerance_matrix: np.ndarray) -> Tuple[FrozenSet[int], ...]:
    return tuple(
        frozenset(int(index) for index in np.flatnonzero(row)) for row in tolerance_matrix
    )


def _is_subset_of_any(candidate: FrozenSet[int], supersets: Iterable[FrozenSet[int]]) -> bool:
    return any(candidate < existing for existing in supersets)


def compute_maximal_consistent_blocks(
    condition_values: Sequence[Sequence[str]],
    *,
    missing_value: str = MISSING_VALUE,
    one_based: bool = False,
) -> List[FrozenSet[int]]:
    """
    Recover maximal consistent blocks described in the paper.

    The implementation follows the matrix-based tolerance relation and deduplicates
    intersections of tolerance classes before keeping only maximal blocks.
    """

    tolerance_matrix = compute_tolerance_matrix(condition_values, missing_value=missing_value)
    tolerance_classes = compute_tolerance_classes(tolerance_matrix)

    candidates: Set[FrozenSet[int]] = set()
    for i, class_i in enumerate(tolerance_classes):
        for j in class_i:
            if j < i:
                continue
            block = frozenset(class_i & tolerance_classes[j])
            if block:
                candidates.add(block)

    maximal_blocks = sorted(
        (
            block
            for block in candidates
            if not _is_subset_of_any(block, candidates)
        ),
        key=lambda block: (len(block), tuple(sorted(block))),
    )

    if one_based:
        return [frozenset(index + 1 for index in block) for block in maximal_blocks]
    return maximal_blocks


def build_block_decision_table(
    table: IncompleteSingleLabelDecisionTable,
    blocks: Sequence[FrozenSet[int]],
    *,
    missing_value: str = MISSING_VALUE,
) -> BlockDecisionTable:
    """Construct the most fully described decision table CDT."""

    block_condition_rows: List[Tuple[str, ...]] = []
    block_decision_sets: List[FrozenSet[str]] = []

    for block in blocks:
        sorted_block = sorted(block)
        row_values: List[str] = []
        for attr_index in range(table.attribute_count):
            observed_value = missing_value
            for object_index in sorted_block:
                value = table.condition_values[object_index][attr_index]
                if value != missing_value:
                    observed_value = value
                    break
            row_values.append(observed_value)

        decision_set = frozenset(
            table.decision_values[object_index] for object_index in sorted_block
        )
        block_condition_rows.append(tuple(row_values))
        block_decision_sets.append(decision_set)

    return BlockDecisionTable(
        condition_attributes=table.condition_attributes,
        condition_values=tuple(block_condition_rows),
        decision_attribute=table.decision_attribute,
        decision_values=tuple(block_decision_sets),
        source_blocks=tuple(blocks),
    )


def compute_discernibility_clauses(block_table: BlockDecisionTable) -> List[int]:
    """
    Build the discernibility clauses from the block decision table.

    Each clause is encoded as an integer bitmask over the condition attributes.
    """

    clauses: Set[int] = set()
    block_count = len(block_table.condition_values)
    attribute_count = len(block_table.condition_attributes)
    for i in range(block_count):
        for j in range(i + 1, block_count):
            if block_table.decision_values[i] == block_table.decision_values[j]:
                continue
            clause = 0
            for attr_index in range(attribute_count):
                if (
                    block_table.condition_values[i][attr_index]
                    != block_table.condition_values[j][attr_index]
                ):
                    clause |= 1 << attr_index
            if clause:
                clauses.add(clause)
    return _prune_clause_masks(sorted(clauses, key=lambda mask: (mask.bit_count(), mask)))


def _prune_clause_masks(clauses: Sequence[int]) -> List[int]:
    unique_clauses = sorted(set(clauses), key=lambda mask: (mask.bit_count(), mask))
    pruned: List[int] = []
    for clause in unique_clauses:
        if any((existing & clause) == existing for existing in pruned):
            continue
        pruned.append(clause)
    return pruned


def _insert_minimal_solution(solutions: List[int], candidate: int) -> None:
    if any((existing & candidate) == existing for existing in solutions):
        return
    solutions[:] = [existing for existing in solutions if not (candidate & existing) == candidate]
    solutions.append(candidate)
    solutions.sort(key=lambda mask: (mask.bit_count(), mask))


def _iter_mask_bits(mask: int) -> List[int]:
    bits: List[int] = []
    remaining = mask
    while remaining:
        bit = remaining & -remaining
        bits.append(bit)
        remaining -= bit
    return bits


def _greedy_hitting_set_mask(clauses: Sequence[int]) -> int:
    mask = 0
    uncovered = list(clauses)
    while uncovered:
        bit_scores: dict[int, int] = {}
        for clause in uncovered:
            for bit in _iter_mask_bits(clause):
                bit_scores[bit] = bit_scores.get(bit, 0) + 1
        best_bit = min(
            bit_scores,
            key=lambda bit: (-bit_scores[bit], bit.bit_length()),
        )
        mask |= best_bit
        uncovered = [clause for clause in uncovered if not (mask & clause)]
    return mask


def _exact_single_reduct_mask(clauses: Sequence[int]) -> int:
    if not clauses:
        return 0

    sorted_clauses = tuple(sorted(set(clauses), key=lambda clause: (clause.bit_count(), clause)))
    best_mask = _greedy_hitting_set_mask(sorted_clauses)
    best_size = best_mask.bit_count()

    def search(current_mask: int) -> None:
        nonlocal best_mask, best_size
        current_size = current_mask.bit_count()
        if current_size >= best_size:
            return

        pivot_clause = None
        for clause in sorted_clauses:
            if not (current_mask & clause):
                pivot_clause = clause
                break

        if pivot_clause is None:
            best_mask = current_mask
            best_size = current_size
            return

        candidate_bits = _iter_mask_bits(pivot_clause)
        coverage_counts = {
            bit: sum(1 for clause in sorted_clauses if not (current_mask & clause) and (clause & bit))
            for bit in candidate_bits
        }
        candidate_bits.sort(key=lambda bit: (-coverage_counts[bit], bit.bit_length()))
        for bit in candidate_bits:
            search(current_mask | bit)

    search(0)
    return best_mask


def _enumerate_minimal_hitting_sets(
    clauses: Sequence[int],
    *,
    max_solutions: Optional[int] = None,
) -> List[int]:
    if not clauses:
        return [0]

    solutions: List[int] = []
    sorted_clauses = tuple(sorted(clauses, key=lambda clause: (clause.bit_count(), clause)))

    def search(current_mask: int) -> None:
        if max_solutions is not None and len(solutions) >= max_solutions:
            return
        if any((existing & current_mask) == existing for existing in solutions):
            return

        uncovered_clause = None
        for clause in sorted_clauses:
            if not (current_mask & clause):
                uncovered_clause = clause
                break

        if uncovered_clause is None:
            _insert_minimal_solution(solutions, current_mask)
            return

        branch_bits = uncovered_clause
        while branch_bits:
            bit = branch_bits & -branch_bits
            branch_bits -= bit
            next_mask = current_mask | bit
            search(next_mask)

    search(0)
    return solutions


def _mask_to_attribute_names(mask: int, attribute_names: Sequence[str]) -> FrozenSet[str]:
    return frozenset(
        attribute_names[index]
        for index in range(len(attribute_names))
        if mask & (1 << index)
    )


def compute_mcb_ar_reducts(
    table: IncompleteSingleLabelDecisionTable,
    *,
    max_reducts: Optional[int] = None,
    missing_value: str = MISSING_VALUE,
) -> List[FrozenSet[str]]:
    blocks = compute_maximal_consistent_blocks(
        table.condition_values,
        missing_value=missing_value,
    )
    block_table = build_block_decision_table(
        table,
        blocks,
        missing_value=missing_value,
    )
    clause_masks = compute_discernibility_clauses(block_table)
    if max_reducts == 1:
        return [_mask_to_attribute_names(_exact_single_reduct_mask(clause_masks), table.condition_attributes)]
    reduct_masks = _enumerate_minimal_hitting_sets(
        clause_masks,
        max_solutions=max_reducts,
    )
    return [
        _mask_to_attribute_names(mask, table.condition_attributes)
        for mask in reduct_masks
    ]


class MCBARReducer:
    """Reference implementation of the MCB-AR attribute-reduction method."""

    def __init__(self, *, missing_value: str = MISSING_VALUE) -> None:
        self.missing_value = missing_value
        self.table_: Optional[IncompleteSingleLabelDecisionTable] = None
        self.blocks_: Tuple[FrozenSet[int], ...] = ()
        self.block_table_: Optional[BlockDecisionTable] = None
        self.reducts_: Tuple[FrozenSet[str], ...] = ()

    def fit(self, table: IncompleteSingleLabelDecisionTable) -> "MCBARReducer":
        self.table_ = table
        self.blocks_ = tuple(
            compute_maximal_consistent_blocks(
                table.condition_values,
                missing_value=self.missing_value,
            )
        )
        self.block_table_ = build_block_decision_table(
            table,
            self.blocks_,
            missing_value=self.missing_value,
        )
        clause_masks = compute_discernibility_clauses(self.block_table_)
        self.reducts_ = tuple(
            _mask_to_attribute_names(mask, table.condition_attributes)
            for mask in _enumerate_minimal_hitting_sets(clause_masks)
        )
        return self

    def shortest_reduct(self) -> FrozenSet[str]:
        if not self.reducts_:
            raise RuntimeError("fit() must be called before requesting a reduct.")
        return min(self.reducts_, key=lambda reduct: (len(reduct), tuple(sorted(reduct))))


__all__ = [
    "BlockDecisionTable",
    "IncompleteSingleLabelDecisionTable",
    "MCBARReducer",
    "MISSING_VALUE",
    "build_block_decision_table",
    "build_incomplete_single_label_table",
    "compute_discernibility_clauses",
    "compute_maximal_consistent_blocks",
    "compute_mcb_ar_reducts",
    "compute_tolerance_classes",
    "compute_tolerance_matrix",
]
