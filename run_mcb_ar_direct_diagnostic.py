from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from algorithm1 import (
    IncompleteMultiLabelDecisionTable,
    _build_block_attribute_cache,
    _build_object_attribute_codes,
    _compute_decision_signature_ids,
    _compute_label_sets,
    _compute_phi_by_object,
    _pair_clause_mask,
    _to_zero_based_blocks,
    algorithm1_maximal_compatibility_blocks,
)
from algorithms.mcb_ar import compute_maximal_consistent_blocks
from arff_parser import read_arff
from experiment_core import (
    BLOCKWISE,
    BY_ATTRIBUTE,
    BY_OBJECT,
    DATA_DIR,
    MCAR,
    MCB_AR,
    OURS,
    build_kfold_indices,
    inject_missing_values,
    matrix_to_incomplete_rows,
    run_comparison_experiment,
    _label_rows_to_single_decisions,
    _sample_dataset,
)


DATASET_PATHS = {
    "birds": DATA_DIR / "birds.arff",
    "scene": DATA_DIR / "scene.arff",
    "yeast": DATA_DIR / "yeast.arff",
}
PATTERN_LABELS = {
    MCAR: "MCAR",
    BY_OBJECT: "Object-wise",
    BY_ATTRIBUTE: "Attribute-wise",
    BLOCKWISE: "Blockwise",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the direct Ours-vs-MCB-AR diagnostic study under structured missingness."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_PATHS),
        default=list(DATASET_PATHS),
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=[MCAR, BY_OBJECT, BY_ATTRIBUTE, BLOCKWISE],
        default=[MCAR, BY_OBJECT, BY_ATTRIBUTE, BLOCKWISE],
    )
    parser.add_argument("--missing-ratio", type=float, default=0.10)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-objects", type=int, default=40)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "mcb_ar_direct_diagnostic_v1",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    left = set(a)
    right = set(b)
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _ours_block_and_omission_stats(
    train_X_missing: np.ndarray,
    train_y: np.ndarray,
    feature_names: Sequence[str],
    label_names: Sequence[str],
) -> Dict[str, float]:
    rows = matrix_to_incomplete_rows(train_X_missing)
    table = IncompleteMultiLabelDecisionTable(
        condition_attributes=tuple(feature_names),
        condition_values=rows,
        label_names=tuple(label_names),
        label_values=train_y.astype(np.int8).tolist(),
    )

    blocks_one_based = algorithm1_maximal_compatibility_blocks(table.condition_values)
    blocks_zero_based = _to_zero_based_blocks(blocks_one_based)
    phi_by_object = _compute_phi_by_object(blocks_zero_based, table.object_count)
    block_missing_all, block_observed_values = _build_block_attribute_cache(
        table.condition_values,
        blocks_zero_based,
        len(feature_names),
    )
    block_to_index = {block: idx for idx, block in enumerate(blocks_zero_based)}
    phi_block_indices_by_object = [
        [block_to_index[block] for block in phi_blocks] for phi_blocks in phi_by_object
    ]
    object_attr_codes = _build_object_attribute_codes(
        phi_block_indices_by_object,
        block_missing_all,
        block_observed_values,
        len(feature_names),
    )

    label_sets = _compute_label_sets(table.label_values, table.label_names)
    block_union_map: Dict[frozenset[int], set[str]] = {}
    block_intersection_map: Dict[frozenset[int], set[str]] = {}
    for block in blocks_zero_based:
        union_labels: set[str] = set()
        intersection_labels: set[str] | None = None
        for obj_idx in block:
            obj_labels = label_sets[obj_idx]
            union_labels.update(obj_labels)
            if intersection_labels is None:
                intersection_labels = set(obj_labels)
            else:
                intersection_labels.intersection_update(obj_labels)
        block_union_map[block] = union_labels
        block_intersection_map[block] = intersection_labels or set()

    coarse_sets = []
    fine_sets = []
    for phi_blocks in phi_by_object:
        coarse: set[str] = set()
        fine: set[str] | None = None
        for block in phi_blocks:
            coarse.update(block_union_map[block])
            block_common = block_intersection_map[block]
            if fine is None:
                fine = set(block_common)
            else:
                fine.intersection_update(block_common)
        coarse_sets.append(frozenset(coarse))
        fine_sets.append(frozenset(fine or set()))

    decision_signature_ids = _compute_decision_signature_ids(coarse_sets, fine_sets)
    attr_bit_lookup = tuple(1 << attr_idx for attr_idx in range(len(feature_names)))

    conflict_pairs = 0
    omitted_pairs = 0
    for i in range(table.object_count):
        for j in range(i + 1, table.object_count):
            if int(decision_signature_ids[i]) == int(decision_signature_ids[j]):
                continue
            conflict_pairs += 1
            if _pair_clause_mask(object_attr_codes, i, j, attr_bit_lookup) == 0:
                omitted_pairs += 1

    return {
        "block_count": float(len(blocks_zero_based)),
        "conflict_pairs": float(conflict_pairs),
        "omitted_pairs": float(omitted_pairs),
        "omission_rate": float(omitted_pairs / conflict_pairs) if conflict_pairs else 0.0,
    }


def _mcb_ar_block_and_omission_stats(
    train_X_missing: np.ndarray,
    train_y: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, float]:
    rows = matrix_to_incomplete_rows(train_X_missing)
    blocks_zero_based = compute_maximal_consistent_blocks(rows)
    phi_by_object = _compute_phi_by_object(blocks_zero_based, len(rows))
    block_missing_all, block_observed_values = _build_block_attribute_cache(
        rows,
        blocks_zero_based,
        len(feature_names),
    )
    block_to_index = {block: idx for idx, block in enumerate(blocks_zero_based)}
    phi_block_indices_by_object = [
        [block_to_index[block] for block in phi_blocks] for phi_blocks in phi_by_object
    ]
    object_attr_codes = _build_object_attribute_codes(
        phi_block_indices_by_object,
        block_missing_all,
        block_observed_values,
        len(feature_names),
    )

    decision_values = _label_rows_to_single_decisions(train_y)
    signature_to_id: Dict[str, int] = {}
    decision_ids = np.empty(len(decision_values), dtype=np.int32)
    for obj_idx, signature in enumerate(decision_values):
        signature_id = signature_to_id.get(signature)
        if signature_id is None:
            signature_id = len(signature_to_id)
            signature_to_id[signature] = signature_id
        decision_ids[obj_idx] = signature_id

    attr_bit_lookup = tuple(1 << attr_idx for attr_idx in range(len(feature_names)))
    conflict_pairs = 0
    omitted_pairs = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if int(decision_ids[i]) == int(decision_ids[j]):
                continue
            conflict_pairs += 1
            if _pair_clause_mask(object_attr_codes, i, j, attr_bit_lookup) == 0:
                omitted_pairs += 1

    return {
        "block_count": float(len(blocks_zero_based)),
        "conflict_pairs": float(conflict_pairs),
        "omitted_pairs": float(omitted_pairs),
        "omission_rate": float(omitted_pairs / conflict_pairs) if conflict_pairs else 0.0,
    }


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def _build_markdown_summary(aggregate_rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Ours vs MCB-AR Direct Diagnostic Study",
        "",
        "All values are reported as mean ± std over the three public datasets and two folds under each missingness protocol.",
        "",
        "| Protocol | Method | Blocks | Omitted pairs | Omission rate | Selected attrs | RR (%) | F1 | RL | Mean reduct Jaccard with Ours |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["missing_pattern_label"]),
                    str(row["method_label"]),
                    f'{row["mean_block_count"]:.3f} $\\pm$ {row["std_block_count"]:.3f}',
                    f'{row["mean_omitted_pairs"]:.3f} $\\pm$ {row["std_omitted_pairs"]:.3f}',
                    f'{row["mean_omission_rate"]:.3f} $\\pm$ {row["std_omission_rate"]:.3f}',
                    f'{row["mean_selected_attributes"]:.3f} $\\pm$ {row["std_selected_attributes"]:.3f}',
                    f'{row["mean_reduction_rate"]:.3f} $\\pm$ {row["std_reduction_rate"]:.3f}',
                    f'{row["mean_f1"]:.3f} $\\pm$ {row["std_f1"]:.3f}',
                    f'{row["mean_ranking_loss"]:.3f} $\\pm$ {row["std_ranking_loss"]:.3f}',
                    (
                        f'{row["mean_reduct_jaccard_with_ours"]:.3f} $\\pm$ {row["std_reduct_jaccard_with_ours"]:.3f}'
                        if row["method"] == MCB_AR
                        else "1.000 $\\pm$ 0.000"
                    ),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _plot_aggregate_rows(rows: Sequence[Dict[str, Any]], output_path: Path) -> Path | None:
    if plt is None or not rows:
        return None

    patterns = [MCAR, BY_OBJECT, BY_ATTRIBUTE, BLOCKWISE]
    methods = [OURS, MCB_AR]
    metrics = (
        ("mean_block_count", "Recovered maximal blocks"),
        ("mean_omission_rate", "Decision-level omission rate"),
        ("mean_f1", "F1-score"),
        ("mean_ranking_loss", "Ranking Loss"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    for axis, (metric_key, title) in zip(axes.ravel(), metrics):
        for method in methods:
            filtered = [row for row in rows if row["method"] == method]
            filtered.sort(key=lambda row: patterns.index(row["missing_pattern"]))
            axis.plot(
                [PATTERN_LABELS[row["missing_pattern"]] for row in filtered],
                [row[metric_key] for row in filtered],
                marker="o",
                linewidth=2,
                label="Ours" if method == OURS else "MCB-AR",
            )
        axis.set_title(title)
        axis.grid(alpha=0.3)
    axes.ravel()[0].legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    raw_payload: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        data_path = DATASET_PATHS[dataset]
        feature_names, feature_rows, label_names, label_rows = read_arff(data_path)
        features = np.asarray(feature_rows, dtype=np.float64)
        labels = np.asarray(label_rows, dtype=np.int8)
        features, labels = _sample_dataset(
            features,
            labels,
            sample_fraction=args.sample_fraction,
            max_objects=args.max_objects,
            seed=args.seed,
        )

        for pattern in args.patterns:
            comparison = run_comparison_experiment(
                data_path=data_path,
                folds=args.folds,
                missing_ratio=args.missing_ratio,
                seed=args.seed,
                algorithms=[OURS, MCB_AR],
                sample_fraction=args.sample_fraction,
                max_objects=args.max_objects,
                max_reducts=1,
                timeout=args.timeout if args.timeout > 0 else None,
                prefer_greedy=False,
                verbose=False,
                missing_pattern=pattern,
            )

            features_missing, _ = inject_missing_values(
                features,
                args.missing_ratio,
                args.seed,
                pattern=pattern,
            )
            folds_indices = build_kfold_indices(len(features), args.folds, args.seed, overlap_ratio=0.0)
            ours_folds = comparison.summaries[OURS].fold_results
            mcb_folds = comparison.summaries[MCB_AR].fold_results

            for fold_idx, (train_indices, _test_indices) in enumerate(folds_indices, start=1):
                train_X_missing = features_missing[train_indices]
                train_y = labels[train_indices]
                ours_diag = _ours_block_and_omission_stats(
                    train_X_missing,
                    train_y,
                    feature_names,
                    label_names,
                )
                mcb_diag = _mcb_ar_block_and_omission_stats(
                    train_X_missing,
                    train_y,
                    feature_names,
                )
                ours_fold = ours_folds[fold_idx - 1]
                mcb_fold = mcb_folds[fold_idx - 1]
                reduct_jaccard = _jaccard(ours_fold.chosen_features, mcb_fold.chosen_features)

                for method, diag, fold in (
                    (OURS, ours_diag, ours_fold),
                    (MCB_AR, mcb_diag, mcb_fold),
                ):
                    fold_rows.append(
                        {
                            "dataset": dataset,
                            "missing_pattern": pattern,
                            "missing_pattern_label": PATTERN_LABELS[pattern],
                            "fold_index": fold_idx,
                            "method": method,
                            "method_label": "Ours" if method == OURS else "MCB-AR",
                            "block_count": diag["block_count"],
                            "conflict_pairs": diag["conflict_pairs"],
                            "omitted_pairs": diag["omitted_pairs"],
                            "omission_rate": diag["omission_rate"],
                            "selected_attributes": fold.selected_attributes,
                            "reduction_rate": fold.reduction_rate,
                            "test_f1_score": fold.test_f1_score,
                            "test_ranking_loss": fold.test_ranking_loss,
                            "chosen_features": ",".join(fold.chosen_features),
                            "reduct_jaccard_with_ours": 1.0 if method == OURS else reduct_jaccard,
                        }
                    )

                raw_payload.append(
                    {
                        "dataset": dataset,
                        "pattern": pattern,
                        "fold_index": fold_idx,
                        "ours": ours_diag,
                        "mcb_ar": mcb_diag,
                        "ours_features": ours_fold.chosen_features,
                        "mcb_ar_features": mcb_fold.chosen_features,
                        "reduct_jaccard": reduct_jaccard,
                        "ours_f1": ours_fold.test_f1_score,
                        "mcb_ar_f1": mcb_fold.test_f1_score,
                        "ours_ranking_loss": ours_fold.test_ranking_loss,
                        "mcb_ar_ranking_loss": mcb_fold.test_ranking_loss,
                    }
                )

    for pattern in args.patterns:
        pattern_rows = [row for row in fold_rows if row["missing_pattern"] == pattern]
        for method in (OURS, MCB_AR):
            method_rows = [row for row in pattern_rows if row["method"] == method]
            if not method_rows:
                continue
            aggregate_rows.append(
                {
                    "missing_pattern": pattern,
                    "missing_pattern_label": PATTERN_LABELS[pattern],
                    "method": method,
                    "method_label": "Ours" if method == OURS else "MCB-AR",
                    "mean_block_count": _mean_std([row["block_count"] for row in method_rows])[0],
                    "std_block_count": _mean_std([row["block_count"] for row in method_rows])[1],
                    "mean_omitted_pairs": _mean_std([row["omitted_pairs"] for row in method_rows])[0],
                    "std_omitted_pairs": _mean_std([row["omitted_pairs"] for row in method_rows])[1],
                    "mean_omission_rate": _mean_std([row["omission_rate"] for row in method_rows])[0],
                    "std_omission_rate": _mean_std([row["omission_rate"] for row in method_rows])[1],
                    "mean_selected_attributes": _mean_std([row["selected_attributes"] for row in method_rows])[0],
                    "std_selected_attributes": _mean_std([row["selected_attributes"] for row in method_rows])[1],
                    "mean_reduction_rate": _mean_std([row["reduction_rate"] for row in method_rows])[0],
                    "std_reduction_rate": _mean_std([row["reduction_rate"] for row in method_rows])[1],
                    "mean_f1": _mean_std([row["test_f1_score"] for row in method_rows])[0],
                    "std_f1": _mean_std([row["test_f1_score"] for row in method_rows])[1],
                    "mean_ranking_loss": _mean_std([row["test_ranking_loss"] for row in method_rows])[0],
                    "std_ranking_loss": _mean_std([row["test_ranking_loss"] for row in method_rows])[1],
                    "mean_reduct_jaccard_with_ours": _mean_std(
                        [row["reduct_jaccard_with_ours"] for row in method_rows]
                    )[0],
                    "std_reduct_jaccard_with_ours": _mean_std(
                        [row["reduct_jaccard_with_ours"] for row in method_rows]
                    )[1],
                }
            )

    _write_csv(output_dir / "fold_rows.csv", fold_rows)
    _write_csv(output_dir / "aggregate.csv", aggregate_rows)
    _write_json(output_dir / "results.json", raw_payload)
    (output_dir / "summary.md").write_text(
        _build_markdown_summary(aggregate_rows),
        encoding="utf-8",
    )
    _plot_aggregate_rows(aggregate_rows, output_dir / "diagnostic_plot.png")


if __name__ == "__main__":
    main()
