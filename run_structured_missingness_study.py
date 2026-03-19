from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from experiment_core import (
    BLOCKWISE,
    BY_ATTRIBUTE,
    BY_OBJECT,
    DATA_DIR,
    MCAR,
    FIMF,
    MCB_AR,
    MFS_MCDM,
    ML_CSFS,
    OURS,
    ComparisonResult,
    ExperimentSummary,
    run_comparison_experiment,
)


DATASET_PATHS = {
    "birds": DATA_DIR / "birds.arff",
    "scene": DATA_DIR / "scene.arff",
    "yeast": DATA_DIR / "yeast.arff",
}
ALGORITHM_LABELS = {
    OURS: "Ours",
    MCB_AR: "MCB-AR",
    MFS_MCDM: "MFS-MCDM",
    FIMF: "FIMF",
    ML_CSFS: "ML-CSFS",
}
PATTERN_LABELS = {
    MCAR: "MCAR",
    BY_OBJECT: "Object-wise",
    BY_ATTRIBUTE: "Attribute-wise",
    BLOCKWISE: "Blockwise",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the structured missingness stress test used in the paper."
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
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=[OURS, MCB_AR, MFS_MCDM, FIMF, ML_CSFS],
        default=[OURS, ML_CSFS, MCB_AR],
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
        default=Path(__file__).resolve().parent / "structured_missingness_study",
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


def _summary_row(dataset: str, summary: ExperimentSummary) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "missing_pattern": summary.missing_pattern,
        "missing_pattern_label": PATTERN_LABELS[summary.missing_pattern],
        "missing_ratio": summary.missing_ratio,
        "algorithm": summary.algorithm,
        "algorithm_label": ALGORITHM_LABELS[summary.algorithm],
        "avg_f1_score": summary.avg_f1_score,
        "std_f1_score": summary.std_f1_score,
        "avg_ranking_loss": summary.avg_ranking_loss,
        "std_ranking_loss": summary.std_ranking_loss,
        "avg_coverage": summary.avg_coverage,
        "std_coverage": summary.std_coverage,
        "avg_hamming_loss": summary.avg_hamming_loss,
        "std_hamming_loss": summary.std_hamming_loss,
        "avg_reduction_rate": summary.avg_reduction_rate,
        "std_reduction_rate": summary.std_reduction_rate,
        "avg_pure_reduction_time": summary.avg_reduction_time,
        "avg_configured_reduction_time": summary.avg_reduction_time + summary.avg_preprocessing_time,
        "avg_full_training_time": summary.avg_selection_pipeline_time,
        "dominant_imputer": summary.imputer,
    }


def _aggregate_rows(task_results: Sequence[ComparisonResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    patterns = [MCAR, BY_OBJECT, BY_ATTRIBUTE, BLOCKWISE]
    for pattern in patterns:
        pattern_tasks = [task for task in task_results if task.missing_pattern == pattern]
        if not pattern_tasks:
            continue
        algorithms = sorted(pattern_tasks[0].summaries, key=lambda name: list(ALGORITHM_LABELS).index(name))
        for algorithm in algorithms:
            summaries = [task.summaries[algorithm] for task in pattern_tasks]
            rows.append(
                {
                    "missing_pattern": pattern,
                    "missing_pattern_label": PATTERN_LABELS[pattern],
                    "algorithm": algorithm,
                    "algorithm_label": ALGORITHM_LABELS[algorithm],
                    "mean_f1_score": statistics.mean(summary.avg_f1_score for summary in summaries),
                    "std_f1_score": statistics.stdev(summary.avg_f1_score for summary in summaries)
                    if len(summaries) > 1
                    else 0.0,
                    "mean_ranking_loss": statistics.mean(summary.avg_ranking_loss for summary in summaries),
                    "std_ranking_loss": statistics.stdev(summary.avg_ranking_loss for summary in summaries)
                    if len(summaries) > 1
                    else 0.0,
                    "mean_reduction_rate": statistics.mean(summary.avg_reduction_rate for summary in summaries),
                    "std_reduction_rate": statistics.stdev(summary.avg_reduction_rate for summary in summaries)
                    if len(summaries) > 1
                    else 0.0,
                    "mean_configured_reduction_time": statistics.mean(
                        summary.avg_reduction_time + summary.avg_preprocessing_time
                        for summary in summaries
                    ),
                    "std_configured_reduction_time": statistics.stdev(
                        summary.avg_reduction_time + summary.avg_preprocessing_time
                        for summary in summaries
                    )
                    if len(summaries) > 1
                    else 0.0,
                }
            )
    return rows


def _plot_rows(rows: Sequence[Dict[str, Any]], output_path: Path) -> Optional[Path]:
    if plt is None or not rows:
        return None

    metrics = (
        ("mean_f1_score", "F1-score", False),
        ("mean_ranking_loss", "Ranking Loss", False),
        ("mean_reduction_rate", "Reduction rate (%)", False),
        ("mean_configured_reduction_time", "Configured reduction time (s)", True),
    )
    patterns = [MCAR, BY_OBJECT, BY_ATTRIBUTE, BLOCKWISE]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    for axis, (metric_key, title, use_log) in zip(axes.ravel(), metrics):
        for algorithm in sorted({row["algorithm"] for row in rows}, key=lambda name: list(ALGORITHM_LABELS).index(name)):
            filtered = [row for row in rows if row["algorithm"] == algorithm]
            filtered.sort(key=lambda row: patterns.index(row["missing_pattern"]))
            axis.plot(
                [PATTERN_LABELS[row["missing_pattern"]] for row in filtered],
                [row[metric_key] for row in filtered],
                marker="o",
                linewidth=2,
                label=ALGORITHM_LABELS[algorithm],
            )
        axis.set_title(title)
        axis.grid(alpha=0.3)
        if use_log:
            axis.set_yscale("log")
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

    results: List[ComparisonResult] = []
    summary_rows: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        for pattern in args.patterns:
            result = run_comparison_experiment(
                data_path=DATASET_PATHS[dataset],
                folds=args.folds,
                missing_ratio=args.missing_ratio,
                seed=args.seed,
                algorithms=args.algorithms,
                sample_fraction=args.sample_fraction,
                max_objects=args.max_objects,
                max_reducts=1,
                timeout=args.timeout if args.timeout > 0 else None,
                prefer_greedy=False,
                verbose=False,
                missing_pattern=pattern,
            )
            results.append(result)
            for algorithm, summary in result.summaries.items():
                summary_rows.append(_summary_row(dataset, summary))

    aggregate_rows = _aggregate_rows(results)
    plot_path = _plot_rows(aggregate_rows, output_dir / "structured_missingness.png")

    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "aggregate.csv", aggregate_rows)
    _write_json(output_dir / "results.json", results)
    _write_json(output_dir / "manifest.json", {"plot_path": str(plot_path) if plot_path else None})

    print(f"Output directory: {output_dir}")
    print(f"Summary rows    : {len(summary_rows)}")
    print(f"Aggregate rows  : {len(aggregate_rows)}")


if __name__ == "__main__":
    main()
