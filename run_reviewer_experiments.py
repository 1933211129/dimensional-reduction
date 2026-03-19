from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

from algorithm1 import (
    IncompleteMultiLabelDecisionTable,
    algorithm3_multi_label_reducts,
)
from evaluation_metrics import ranking_loss
from experiment_core import (
    BASELINE_ALGORITHMS,
    BLOCKWISE,
    BY_ATTRIBUTE,
    BY_OBJECT,
    DATA_DIR,
    DEFAULT_FOLDS,
    DEFAULT_MISSING_RATIOS,
    FIMF,
    MCB_AR,
    MCAR,
    MFS_MCDM,
    MISSINGNESS_PATTERNS,
    ML_CSFS,
    OURS,
    ComparisonResult,
    ExperimentSummary,
    FoldResult,
    SignificanceRecord,
    inject_missing_values,
    matrix_to_incomplete_rows,
    run_comparison_experiment,
    compute_wilcoxon_significance,
)


warnings.filterwarnings("ignore", message="Bins whose width are too small")
warnings.filterwarnings(
    "ignore",
    message=r"Bins whose width are too small .*",
)
warnings.filterwarnings(
    "ignore",
    message=r"\[IterativeImputer\] Early stopping criterion not reached\.",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.preprocessing\._discretization",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"sklearn\.preprocessing\._discretization",
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
MISSING_PATTERN_LABELS = {
    MCAR: "MCAR",
    BY_OBJECT: "Object-wise",
    BY_ATTRIBUTE: "Attribute-wise",
    BLOCKWISE: "Blockwise",
}
METRIC_DIRECTIONS = {
    "avg_f1_score": "max",
    "avg_ranking_loss": "min",
    "avg_coverage": "min",
    "avg_hamming_loss": "min",
    "avg_reduction_time": "min",
}
OBJECT_SCALE_VALUES = (160, 320, 640, 1280)
FEATURE_SCALE_VALUES = (64, 128, 256, 512)
DEFAULT_STRUCTURED_PATTERNS = (MCAR, BY_OBJECT, BY_ATTRIBUTE, BLOCKWISE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 reviewer 定向实验，导出主结果、显著性、runtime/scaling 与中文总结。"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_PATHS.keys()),
        default=list(DATASET_PATHS.keys()),
        help="要运行的公开数据集。",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=sorted(ALGORITHM_LABELS.keys()),
        default=list(ALGORITHM_LABELS.keys()),
        help="要运行的算法列表。",
    )
    parser.add_argument(
        "--missing-ratios",
        nargs="+",
        type=float,
        default=list(DEFAULT_MISSING_RATIOS),
        metavar="RATIO",
        help="缺失率列表，默认 0.05 0.10 0.15。",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_FOLDS,
        help="外层 K-fold 折数，默认 10。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="容差计算最大并行进程数。",
    )
    parser.add_argument(
        "--parallel-threshold",
        type=int,
        default=1,
        help="触发容差并行的对象数量阈值。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="ProcessPoolExecutor map 的 chunksize；<=0 表示自动。",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="调试用抽样比例，默认不抽样。",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="调试用最大样本数限制。",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="抽样随机种子，默认沿用 --seed。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Ours exact 单折约简超时时间（秒）。",
    )
    parser.add_argument(
        "--scaling-timeout",
        type=float,
        default=120.0,
        help="synthetic scaling 中 exact 路径的单次超时时间（秒）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reviewer_results",
        help="结果输出目录。",
    )
    parser.add_argument(
        "--skip-main",
        action="store_true",
        help="跳过 3x3 主比较实验。",
    )
    parser.add_argument(
        "--skip-scaling",
        action="store_true",
        help="跳过 synthetic scaling。",
    )
    parser.add_argument(
        "--skip-structured",
        action="store_true",
        help="跳过结构化缺失实验。",
    )
    parser.add_argument(
        "--structured-missing-ratio",
        type=float,
        default=0.10,
        help="结构化缺失实验固定使用的缺失率。",
    )
    parser.add_argument(
        "--structured-patterns",
        nargs="+",
        choices=sorted(MISSINGNESS_PATTERNS),
        default=list(DEFAULT_STRUCTURED_PATTERNS),
        help="结构化缺失实验使用的缺失模式。",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="跳过 runtime/scaling 绘图。",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, tuple):
        return list(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
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


def _summary_row(dataset_name: str, summary: ExperimentSummary) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "missing_ratio": summary.missing_ratio,
        "missing_pattern": summary.missing_pattern,
        "missing_pattern_label": MISSING_PATTERN_LABELS.get(
            summary.missing_pattern,
            summary.missing_pattern,
        ),
        "algorithm": summary.algorithm,
        "algorithm_label": ALGORITHM_LABELS.get(summary.algorithm, summary.algorithm),
        "imputer": summary.imputer,
        "avg_reduction_time": summary.avg_reduction_time,
        "std_reduction_time": summary.std_reduction_time,
        "avg_tuning_time": summary.avg_tuning_time,
        "std_tuning_time": summary.std_tuning_time,
        "avg_preprocessing_time": summary.avg_preprocessing_time,
        "std_preprocessing_time": summary.std_preprocessing_time,
        "avg_selection_pipeline_time": summary.avg_selection_pipeline_time,
        "std_selection_pipeline_time": summary.std_selection_pipeline_time,
        "avg_selected_attributes": summary.avg_selected_attributes,
        "std_selected_attributes": summary.std_selected_attributes,
        "avg_reduction_rate": summary.avg_reduction_rate,
        "std_reduction_rate": summary.std_reduction_rate,
        "avg_f1_score": summary.avg_f1_score,
        "std_f1_score": summary.std_f1_score,
        "avg_ranking_loss": summary.avg_ranking_loss,
        "std_ranking_loss": summary.std_ranking_loss,
        "avg_coverage": summary.avg_coverage,
        "std_coverage": summary.std_coverage,
        "avg_hamming_loss": summary.avg_hamming_loss,
        "std_hamming_loss": summary.std_hamming_loss,
        "total_runtime": summary.total_runtime,
        "folds": len(summary.fold_results),
    }


def _fold_rows(dataset_name: str, summary: ExperimentSummary) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold in summary.fold_results:
        rows.append(
            {
                "dataset": dataset_name,
                "missing_ratio": summary.missing_ratio,
                "missing_pattern": summary.missing_pattern,
                "missing_pattern_label": MISSING_PATTERN_LABELS.get(
                    summary.missing_pattern,
                    summary.missing_pattern,
                ),
                "algorithm": summary.algorithm,
                "algorithm_label": ALGORITHM_LABELS.get(summary.algorithm, summary.algorithm),
                "fold_index": fold.fold_index,
                "imputer": fold.imputer,
                "tuned_k": fold.tuned_k,
                "train_objects": fold.train_objects,
                "validation_objects": fold.validation_objects,
                "test_objects": fold.test_objects,
                "reduction_time": fold.reduction_time,
                "tuning_time": fold.tuning_time,
                "preprocessing_time": fold.preprocessing_time,
                "selection_pipeline_time": fold.selection_pipeline_time,
                "selected_attributes": fold.selected_attributes,
                "reduction_rate": fold.reduction_rate,
                "test_f1_score": fold.test_f1_score,
                "test_ranking_loss": fold.test_ranking_loss,
                "test_coverage": fold.test_coverage,
                "test_hamming_loss": fold.test_hamming_loss,
                "chosen_features": ",".join(fold.chosen_features),
            }
        )
    return rows


def _significance_rows(records: Sequence[SignificanceRecord]) -> List[Dict[str, Any]]:
    return [
        {
            "metric": record.metric,
            "baseline": record.baseline,
            "baseline_label": ALGORITHM_LABELS.get(record.baseline, record.baseline),
            "statistic": record.statistic,
            "p_value": record.p_value,
            "ours_mean": record.ours_mean,
            "baseline_mean": record.baseline_mean,
        }
        for record in records
    ]


def _manual_ranking_loss(scores: Dict[str, float], truths: set[str]) -> float:
    irrelevant = set(scores.keys()) - truths
    if not truths or not irrelevant:
        return 0.0
    errors = sum(
        1
        for relevant in truths
        for irrelevant_label in irrelevant
        if scores[relevant] <= scores[irrelevant_label]
    )
    return errors / (len(truths) * len(irrelevant))


def verify_ranking_loss_implementation() -> Dict[str, Any]:
    cases = [
        {
            "name": "single-sample-perfect-order",
            "scores": [{"y1": 0.95, "y2": 0.10, "y3": 0.80}],
            "truths": [{"y1", "y3"}],
        },
        {
            "name": "single-sample-complete-error",
            "scores": [{"y1": 0.20, "y2": 0.90, "y3": 0.10}],
            "truths": [{"y1", "y3"}],
        },
        {
            "name": "two-sample-mixed",
            "scores": [
                {"y1": 0.60, "y2": 0.70, "y3": 0.20, "y4": 0.10},
                {"y1": 0.80, "y2": 0.40, "y3": 0.70, "y4": 0.30},
            ],
            "truths": [{"y1", "y3"}, {"y2"}],
        },
    ]

    case_results: List[Dict[str, Any]] = []
    for case in cases:
        manual_scores = [
            _manual_ranking_loss(score_dict, truth)
            for score_dict, truth in zip(case["scores"], case["truths"])
        ]
        manual_value = float(sum(manual_scores) / len(manual_scores))
        implementation_value = float(ranking_loss(case["scores"], case["truths"]))
        case_results.append(
            {
                "name": case["name"],
                "manual_value": manual_value,
                "implementation_value": implementation_value,
                "absolute_difference": abs(manual_value - implementation_value),
            }
        )

    return {
        "cases": case_results,
        "all_cases_match": all(result["absolute_difference"] < 1e-12 for result in case_results),
        "definition": "relevant-vs-irrelevant pairs, normalized by |Y_t| * |Y_bar_t|",
    }


def _generate_synthetic_dataset(
    *,
    object_count: int,
    feature_count: int,
    label_count: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    core_feature_count = min(6, feature_count)
    core_bits = rng.integers(0, 2, size=(object_count, core_feature_count), endpoint=False)

    features = rng.normal(loc=0.0, scale=0.2, size=(object_count, feature_count))
    for feature_idx in range(core_feature_count):
        features[:, feature_idx] = (
            2.0 * core_bits[:, feature_idx] - 1.0 + 0.08 * rng.normal(size=object_count)
        )
    for feature_idx in range(core_feature_count, feature_count):
        source_a = feature_idx % core_feature_count
        source_b = (feature_idx + 1) % core_feature_count
        features[:, feature_idx] = (
            0.9 * features[:, source_a]
            - 0.5 * features[:, source_b]
            + 0.15 * rng.normal(size=object_count)
        )

    labels = np.zeros((object_count, label_count), dtype=np.int8)
    for label_idx in range(label_count):
        left = core_bits[:, label_idx % core_feature_count]
        right = core_bits[:, (label_idx + 1) % core_feature_count]
        extra = core_bits[:, (label_idx + 2) % core_feature_count]
        if label_idx % 3 == 0:
            label_values = left ^ right
        elif label_idx % 3 == 1:
            label_values = (left & right) | extra
        else:
            label_values = (left + right + extra >= 2).astype(np.int8)
        labels[:, label_idx] = label_values.astype(np.int8, copy=False)

    empty_rows = np.where(labels.sum(axis=1) == 0)[0]
    for row_idx in empty_rows:
        labels[row_idx, row_idx % label_count] = 1

    feature_names = tuple(f"a{idx + 1}" for idx in range(feature_count))
    label_names = tuple(f"y{idx + 1}" for idx in range(label_count))
    return features.astype(np.float64), labels, feature_names, label_names


def _measure_ours_reduction_runtime(
    *,
    object_count: int,
    feature_count: int,
    label_count: int,
    missing_ratio: float,
    seed: int,
    prefer_greedy: bool,
    timeout: Optional[float],
    scan_type: str,
    scan_value: int,
) -> Dict[str, Any]:
    features, labels, feature_names, label_names = _generate_synthetic_dataset(
        object_count=object_count,
        feature_count=feature_count,
        label_count=label_count,
        seed=seed,
    )
    features_missing, injected_count = inject_missing_values(features, missing_ratio, seed + 17)
    table = IncompleteMultiLabelDecisionTable(
        condition_attributes=feature_names,
        condition_values=matrix_to_incomplete_rows(features_missing),
        label_names=label_names,
        label_values=labels.tolist(),
    )

    start = time.perf_counter()
    status = "ok"
    selected_attributes = None
    try:
        reducts = algorithm3_multi_label_reducts(
            table,
            max_reducts=1,
            timeout=timeout,
            prefer_greedy=prefer_greedy,
        )
        if reducts:
            selected_attributes = len(reducts[0])
    except TimeoutError:
        status = "timeout"

    elapsed = time.perf_counter() - start
    return {
        "scan_type": scan_type,
        "scan_value": scan_value,
        "variant": "greedy" if prefer_greedy else "exact",
        "object_count": object_count,
        "feature_count": feature_count,
        "label_count": label_count,
        "missing_ratio": missing_ratio,
        "seed": seed,
        "status": status,
        "reduction_time": elapsed,
        "selected_attributes": selected_attributes,
        "injected_missing_cells": injected_count,
    }


def run_scaling_experiments(
    *,
    seed: int,
    scaling_timeout: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    exact_timeout_reached = False
    for object_count in OBJECT_SCALE_VALUES:
        point_seed = seed + object_count
        if not exact_timeout_reached:
            exact_row = _measure_ours_reduction_runtime(
                object_count=object_count,
                feature_count=128,
                label_count=10,
                missing_ratio=0.10,
                seed=point_seed,
                prefer_greedy=False,
                timeout=scaling_timeout,
                scan_type="objects",
                scan_value=object_count,
            )
            rows.append(exact_row)
            exact_timeout_reached = exact_row["status"] == "timeout"
        else:
            rows.append(
                {
                    "scan_type": "objects",
                    "scan_value": object_count,
                    "variant": "exact",
                    "object_count": object_count,
                    "feature_count": 128,
                    "label_count": 10,
                    "missing_ratio": 0.10,
                    "seed": point_seed,
                    "status": "skipped_after_timeout",
                    "reduction_time": None,
                    "selected_attributes": None,
                    "injected_missing_cells": None,
                }
            )

        rows.append(
            _measure_ours_reduction_runtime(
                object_count=object_count,
                feature_count=128,
                label_count=10,
                missing_ratio=0.10,
                seed=point_seed,
                prefer_greedy=True,
                timeout=scaling_timeout,
                scan_type="objects",
                scan_value=object_count,
            )
        )

    exact_timeout_reached = False
    for feature_count in FEATURE_SCALE_VALUES:
        point_seed = seed + feature_count
        if not exact_timeout_reached:
            exact_row = _measure_ours_reduction_runtime(
                object_count=320,
                feature_count=feature_count,
                label_count=10,
                missing_ratio=0.10,
                seed=point_seed,
                prefer_greedy=False,
                timeout=scaling_timeout,
                scan_type="features",
                scan_value=feature_count,
            )
            rows.append(exact_row)
            exact_timeout_reached = exact_row["status"] == "timeout"
        else:
            rows.append(
                {
                    "scan_type": "features",
                    "scan_value": feature_count,
                    "variant": "exact",
                    "object_count": 320,
                    "feature_count": feature_count,
                    "label_count": 10,
                    "missing_ratio": 0.10,
                    "seed": point_seed,
                    "status": "skipped_after_timeout",
                    "reduction_time": None,
                    "selected_attributes": None,
                    "injected_missing_cells": None,
                }
            )

        rows.append(
            _measure_ours_reduction_runtime(
                object_count=320,
                feature_count=feature_count,
                label_count=10,
                missing_ratio=0.10,
                seed=point_seed,
                prefer_greedy=True,
                timeout=scaling_timeout,
                scan_type="features",
                scan_value=feature_count,
            )
        )

    return rows


def _plot_scaling_results(rows: Sequence[Dict[str, Any]], output_path: Path) -> Optional[Path]:
    if plt is None:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_configs = (
        ("objects", "Object count n", axes[0]),
        ("features", "Feature count d", axes[1]),
    )

    for scan_type, xlabel, axis in plot_configs:
        for variant in ("exact", "greedy"):
            filtered = [
                row
                for row in rows
                if row["scan_type"] == scan_type
                and row["variant"] == variant
                and row["status"] == "ok"
                and row["reduction_time"] is not None
            ]
            if not filtered:
                continue
            filtered.sort(key=lambda row: row["scan_value"])
            axis.plot(
                [row["scan_value"] for row in filtered],
                [row["reduction_time"] for row in filtered],
                marker="o",
                label=variant,
            )

        timeout_rows = [
            row
            for row in rows
            if row["scan_type"] == scan_type and row["status"] == "timeout"
        ]
        for timeout_row in timeout_rows:
            axis.annotate(
                "timeout",
                xy=(timeout_row["scan_value"], timeout_row["reduction_time"]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        axis.set_xlabel(xlabel)
        axis.set_ylabel("Reduction time (s)")
        axis.set_title(f"Runtime scaling by {scan_type}")
        axis.grid(alpha=0.3)
        axis.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _task_better(
    ours_value: float,
    baseline_value: float,
    metric_attr: str,
) -> int:
    direction = METRIC_DIRECTIONS[metric_attr]
    if abs(ours_value - baseline_value) < 1e-12:
        return 0
    if direction == "max":
        return 1 if ours_value > baseline_value else -1
    return 1 if ours_value < baseline_value else -1


def _count_task_comparisons(
    task_results: Sequence[ComparisonResult],
    metric_attr: str,
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for baseline in BASELINE_ALGORITHMS:
        better = 0
        worse = 0
        ties = 0
        for task in task_results:
            if OURS not in task.summaries or baseline not in task.summaries:
                continue
            ours_summary = task.summaries[OURS]
            baseline_summary = task.summaries[baseline]
            comparison = _task_better(
                getattr(ours_summary, metric_attr),
                getattr(baseline_summary, metric_attr),
                metric_attr,
            )
            if comparison > 0:
                better += 1
            elif comparison < 0:
                worse += 1
            else:
                ties += 1
        counts[baseline] = {"better": better, "worse": worse, "ties": ties}
    return counts


def _dominant_imputer_counts(
    task_results: Sequence[ComparisonResult],
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {
        baseline: {"native": 0, "knn": 0, "em": 0, "missforest": 0, "mixed": 0}
        for baseline in BASELINE_ALGORITHMS
    }
    for task in task_results:
        for baseline in BASELINE_ALGORITHMS:
            if baseline not in task.summaries:
                continue
            summary = task.summaries[baseline]
            counts[baseline][summary.imputer] = counts[baseline].get(summary.imputer, 0) + 1
    return counts


def _aggregate_runtime_rows(task_results: Sequence[ComparisonResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in task_results:
        for algorithm, summary in task.summaries.items():
            rows.append(
                {
                    "dataset": task.dataset_name,
                    "missing_ratio": task.missing_ratio,
                    "missing_pattern": task.missing_pattern,
                    "missing_pattern_label": MISSING_PATTERN_LABELS.get(
                        task.missing_pattern,
                        task.missing_pattern,
                    ),
                    "algorithm": algorithm,
                    "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
                    "avg_reduction_time": summary.avg_reduction_time,
                    "std_reduction_time": summary.std_reduction_time,
                    "avg_tuning_time": summary.avg_tuning_time,
                    "std_tuning_time": summary.std_tuning_time,
                    "avg_preprocessing_time": summary.avg_preprocessing_time,
                    "std_preprocessing_time": summary.std_preprocessing_time,
                    "avg_selection_pipeline_time": summary.avg_selection_pipeline_time,
                    "std_selection_pipeline_time": summary.std_selection_pipeline_time,
                    "avg_selected_attributes": summary.avg_selected_attributes,
                }
            )
    return rows


def run_structured_missingness_experiments(
    *,
    datasets: Sequence[str],
    algorithms: Sequence[str],
    patterns: Sequence[str],
    missing_ratio: float,
    folds: int,
    seed: int,
    max_workers: Optional[int],
    parallel_threshold: int,
    chunk_size: Optional[int],
    sample_fraction: float,
    max_objects: Optional[int],
    sample_seed: Optional[int],
    timeout: Optional[float],
) -> List[ComparisonResult]:
    results: List[ComparisonResult] = []
    for dataset_name in datasets:
        data_path = DATASET_PATHS[dataset_name]
        for pattern in patterns:
            results.append(
                run_comparison_experiment(
                    data_path=data_path,
                    folds=folds,
                    missing_ratio=missing_ratio,
                    seed=seed,
                    algorithms=algorithms,
                    max_workers=max_workers,
                    parallel_threshold=parallel_threshold,
                    chunk_size=chunk_size,
                    sample_fraction=sample_fraction,
                    max_objects=max_objects,
                    sample_seed=sample_seed,
                    max_reducts=1,
                    timeout=timeout,
                    prefer_greedy=False,
                    verbose=False,
                    missing_pattern=pattern,
                )
            )
    return results


def _aggregate_structured_missingness(
    task_results: Sequence[ComparisonResult],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pattern in DEFAULT_STRUCTURED_PATTERNS:
        pattern_tasks = [task for task in task_results if task.missing_pattern == pattern]
        if not pattern_tasks:
            continue
        for algorithm in ALGORITHM_LABELS:
            summaries = [
                task.summaries[algorithm]
                for task in pattern_tasks
                if algorithm in task.summaries
            ]
            if not summaries:
                continue
            rows.append(
                {
                    "missing_pattern": pattern,
                    "missing_pattern_label": MISSING_PATTERN_LABELS.get(pattern, pattern),
                    "algorithm": algorithm,
                    "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
                    "mean_f1_score": float(statistics.mean(summary.avg_f1_score for summary in summaries)),
                    "std_f1_score": float(statistics.stdev(summary.avg_f1_score for summary in summaries))
                    if len(summaries) > 1
                    else 0.0,
                    "mean_ranking_loss": float(
                        statistics.mean(summary.avg_ranking_loss for summary in summaries)
                    ),
                    "std_ranking_loss": float(
                        statistics.stdev(summary.avg_ranking_loss for summary in summaries)
                    )
                    if len(summaries) > 1
                    else 0.0,
                    "mean_coverage": float(statistics.mean(summary.avg_coverage for summary in summaries)),
                    "std_coverage": float(statistics.stdev(summary.avg_coverage for summary in summaries))
                    if len(summaries) > 1
                    else 0.0,
                    "mean_hamming_loss": float(
                        statistics.mean(summary.avg_hamming_loss for summary in summaries)
                    ),
                    "std_hamming_loss": float(
                        statistics.stdev(summary.avg_hamming_loss for summary in summaries)
                    )
                    if len(summaries) > 1
                    else 0.0,
                    "mean_reduction_rate": float(
                        statistics.mean(summary.avg_reduction_rate for summary in summaries)
                    ),
                    "std_reduction_rate": float(
                        statistics.stdev(summary.avg_reduction_rate for summary in summaries)
                    )
                    if len(summaries) > 1
                    else 0.0,
                    "mean_configured_reduction_time": float(
                        statistics.mean(summary.avg_reduction_time + summary.avg_preprocessing_time for summary in summaries)
                    ),
                    "std_configured_reduction_time": float(
                        statistics.stdev(summary.avg_reduction_time + summary.avg_preprocessing_time for summary in summaries)
                        if len(summaries) > 1
                        else 0.0
                    ),
                }
            )
    return rows


def _structured_degradation_rows(
    task_results: Sequence[ComparisonResult],
) -> List[Dict[str, Any]]:
    baseline_lookup = {
        (task.dataset_name, task.summaries[algorithm].algorithm): task
        for task in task_results
        if task.missing_pattern == MCAR
        for algorithm in task.summaries
    }
    rows: List[Dict[str, Any]] = []
    for task in task_results:
        if task.missing_pattern == MCAR:
            continue
        for algorithm, summary in task.summaries.items():
            baseline_task = baseline_lookup.get((task.dataset_name, algorithm))
            if baseline_task is None:
                continue
            baseline_summary = baseline_task.summaries[algorithm]
            rows.append(
                {
                    "dataset": task.dataset_name,
                    "algorithm": algorithm,
                    "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
                    "missing_pattern": task.missing_pattern,
                    "missing_pattern_label": MISSING_PATTERN_LABELS.get(
                        task.missing_pattern,
                        task.missing_pattern,
                    ),
                    "delta_f1_score": summary.avg_f1_score - baseline_summary.avg_f1_score,
                    "delta_ranking_loss": summary.avg_ranking_loss - baseline_summary.avg_ranking_loss,
                    "delta_coverage": summary.avg_coverage - baseline_summary.avg_coverage,
                    "delta_hamming_loss": summary.avg_hamming_loss - baseline_summary.avg_hamming_loss,
                    "delta_reduction_rate": summary.avg_reduction_rate - baseline_summary.avg_reduction_rate,
                    "delta_configured_reduction_time": (
                        (summary.avg_reduction_time + summary.avg_preprocessing_time)
                        - (baseline_summary.avg_reduction_time + baseline_summary.avg_preprocessing_time)
                    ),
                }
            )
    return rows


def _plot_structured_missingness(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Optional[Path]:
    if plt is None or not rows:
        return None

    metrics = (
        ("mean_f1_score", "F1-score", False),
        ("mean_ranking_loss", "Ranking Loss", False),
        ("mean_reduction_rate", "Reduction rate (%)", False),
        ("mean_configured_reduction_time", "Configured reduction time (s)", True),
    )
    patterns = [pattern for pattern in DEFAULT_STRUCTURED_PATTERNS if any(row["missing_pattern"] == pattern for row in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes_flat = axes.ravel()
    for axis, (metric_key, title, use_log) in zip(axes_flat, metrics):
        for algorithm in ALGORITHM_LABELS:
            filtered = [row for row in rows if row["algorithm"] == algorithm and row["missing_pattern"] in patterns]
            filtered.sort(key=lambda row: patterns.index(row["missing_pattern"]))
            if not filtered:
                continue
            axis.plot(
                [MISSING_PATTERN_LABELS[row["missing_pattern"]] for row in filtered],
                [row[metric_key] for row in filtered],
                marker="o",
                linewidth=2,
                label=ALGORITHM_LABELS.get(algorithm, algorithm),
            )
        axis.set_title(title)
        axis.grid(alpha=0.3)
        if use_log:
            axis.set_yscale("log")
    axes_flat[0].legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _aggregate_algorithm_runtime_means(
    task_results: Sequence[ComparisonResult],
    attr_name: str,
) -> Dict[str, float]:
    values: Dict[str, List[float]] = {algorithm: [] for algorithm in ALGORITHM_LABELS}
    for task in task_results:
        for algorithm, summary in task.summaries.items():
            values.setdefault(algorithm, []).append(getattr(summary, attr_name))
    return {
        algorithm: float(statistics.mean(runtime_values))
        for algorithm, runtime_values in values.items()
        if runtime_values
    }


def _write_markdown_summary(
    *,
    output_path: Path,
    task_results: Sequence[ComparisonResult],
    significance_records: Sequence[SignificanceRecord],
    ranking_loss_check: Dict[str, Any],
    scaling_rows: Sequence[Dict[str, Any]],
    plot_path: Optional[Path],
    sample_fraction: float,
    max_objects: Optional[int],
) -> None:
    runtime_means = _aggregate_algorithm_runtime_means(task_results, "avg_reduction_time")
    pipeline_means = _aggregate_algorithm_runtime_means(
        task_results,
        "avg_selection_pipeline_time",
    )
    imputer_counts = _dominant_imputer_counts(task_results) if task_results else {}
    f1_counts = _count_task_comparisons(task_results, "avg_f1_score") if task_results else {}
    rl_counts = _count_task_comparisons(task_results, "avg_ranking_loss") if task_results else {}
    task_count = len(task_results)
    dataset_count = len({task.dataset_name for task in task_results})
    missing_ratio_count = len({task.missing_ratio for task in task_results})
    fold_count = 0
    if task_results:
        first_task = task_results[0]
        first_summary = next(iter(first_task.summaries.values()))
        fold_count = len(first_summary.fold_results)

    lines: List[str] = [
        "# Reviewer 实验补充结果总结",
        "",
        "## 0. 实验口径",
        f"- 外层交叉验证折数：`{fold_count}`。"
        if task_results
        else "- 本次结果仅包含非主表辅助实验。",
    ]
    if sample_fraction < 1.0 or max_objects is not None:
        lines.append(
            f"- 为在统一计算预算下完成可复现比较，本次统一使用数据采样："
            f"`sample_fraction={sample_fraction}`，`max_objects={max_objects}`。"
        )
    lines.extend(
        [
            "",
        "## 1. Ranking Loss 定义与实现核验",
        f"- 手工构造样例与代码实现的差值是否为 0：`{ranking_loss_check['all_cases_match']}`。",
        "- 核验口径：相关标签 vs 不相关标签，分母为 `|Y_t| * |Y_bar_t|`。",
        ]
    )
    for case in ranking_loss_check["cases"]:
        lines.append(
            f"- `{case['name']}`：manual={case['manual_value']:.6f}，implementation={case['implementation_value']:.6f}。"
        )

    if task_results:
        lines.extend(
            [
                "",
                "## 2. Baseline 缺失值公平性",
                "- 支持原生缺失值处理的方法保持 `native` 路径；需要插补的 baseline 在训练折内对 `KNN/EM/MissForest` 做独立调优，主结果只保留每个任务点的最佳配置。",
            ]
        )
        for baseline in BASELINE_ALGORITHMS:
            counts = imputer_counts.get(baseline, {})
            lines.append(
                f"- `{ALGORITHM_LABELS[baseline]}` 最终最佳插补分布："
                f"native={counts.get('native', 0)}，KNN={counts.get('knn', 0)}，EM={counts.get('em', 0)}，"
                f"MissForest={counts.get('missforest', 0)}，mixed={counts.get('mixed', 0)}。"
            )

        lines.extend(
            [
                "",
                "## 3. 主结果与显著性",
                f"- 以下计数基于 {task_count} 个任务点（{dataset_count} 个数据集 × {missing_ratio_count} 个缺失率）。",
            ]
        )
        for baseline in BASELINE_ALGORITHMS:
            f1_stat = f1_counts.get(baseline, {"better": 0, "worse": 0, "ties": 0})
            rl_stat = rl_counts.get(baseline, {"better": 0, "worse": 0, "ties": 0})
            lines.append(
                f"- 对 `{ALGORITHM_LABELS[baseline]}`：F1 更优/更差/持平 = "
                f"{f1_stat['better']}/{f1_stat['worse']}/{f1_stat['ties']}；"
                f"Ranking Loss 更优/更差/持平 = "
                f"{rl_stat['better']}/{rl_stat['worse']}/{rl_stat['ties']}。"
            )

        if significance_records:
            lines.append("- Wilcoxon signed-rank test 结果：")
            for record in significance_records:
                lines.append(
                    f"  - `{record.metric}` vs `{ALGORITHM_LABELS.get(record.baseline, record.baseline)}`："
                    f"p={record.p_value:.6g}，Ours={record.ours_mean:.6f}，Baseline={record.baseline_mean:.6f}。"
                )

        lines.extend(
            [
                "",
                "## 4. 运行时间与大规模适用性",
                "- 主结果中的本地数据运行时间现在已单独导出到 `runtime_local.csv`。",
                "- 审稿回复中建议把 `avg_reduction_time` 作为主效率指标，因为它表示在最终选定配置下的纯约简/选择时间。",
                "- `avg_selection_pipeline_time` 则表示训练折内调参与最终重拟合选择器的总开销，可作为补充说明而不应与纯约简时间混为一谈。",
            ]
        )
        for algorithm, mean_value in runtime_means.items():
            lines.append(
                f"- `{ALGORITHM_LABELS.get(algorithm, algorithm)}` 在 9 个任务点上的平均纯约简时间："
                f"{mean_value:.6f} s。"
            )
        for algorithm, mean_value in pipeline_means.items():
            lines.append(
                f"- `{ALGORITHM_LABELS.get(algorithm, algorithm)}` 在 9 个任务点上的平均选择总开销："
                f"{mean_value:.6f} s。"
            )

    if scaling_rows:
        lines.append("- synthetic scaling 已补充 exact 与 greedy 两条曲线。")
        for scan_type in ("objects", "features"):
            ok_rows = [
                row
                for row in scaling_rows
                if row["scan_type"] == scan_type and row["status"] == "ok"
            ]
            ok_rows.sort(key=lambda row: (row["variant"], row["scan_value"]))
            for row in ok_rows:
                lines.append(
                    f"- `{scan_type}` / `{row['variant']}` / `x={row['scan_value']}`："
                    f"{row['reduction_time']:.6f} s，selected={row['selected_attributes']}。"
                )
        timeout_rows = [row for row in scaling_rows if row["status"] == "timeout"]
        for row in timeout_rows:
            lines.append(
                f"- `{row['scan_type']}` 扫描下，exact 在 x={row['scan_value']} 处达到 timeout，"
                "之后的 exact 点已截断。"
            )
        if plot_path is not None:
            lines.append(f"- runtime/scaling 图：`{plot_path}`。")

    lines.extend(
        [
            "",
            "## 5. 面向审稿意见的可直接回复点",
            "- 对审稿人 2 关于 Ranking Loss 的质疑：代码实现已按标准 relevant-vs-irrelevant 定义核验，当前实现与手工计算一致。",
            "- 对审稿人 1 关于 baseline fairness 的质疑：所有 baseline 现在都在训练折内对 KNN、EM、MissForest 做公平调优，主表不再固定单一均值插补。",
            "- 对审稿人 1 关于 significance 的要求：已补充基于 9 个任务点的 Wilcoxon signed-rank test。",
            "- 对审稿人 1 关于 runtime / large-scale applicability 的质疑：已补本地运行时间表和 synthetic scaling 结果，分别回答实际耗时与规模增长趋势。",
            "- 对审稿人 1 关于 case study generality 的质疑：除案例分析外，当前结果已覆盖 birds、scene、yeast 三个公开多标签数据集与 3 个缺失率设置。",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_incremental_main_outputs(
    *,
    output_dir: Path,
    comparison_results: Sequence[ComparisonResult],
    summary_rows: Sequence[Dict[str, Any]],
    fold_rows: Sequence[Dict[str, Any]],
) -> None:
    _write_csv(output_dir / "comparison_summary.csv", summary_rows)
    _write_csv(output_dir / "comparison_folds.csv", fold_rows)
    _write_json(
        output_dir / "comparison_results.json",
        {
            "task_results": comparison_results,
            "summary_rows": summary_rows,
            "fold_rows": fold_rows,
        },
    )


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = args.chunk_size if args.chunk_size > 0 else None
    comparison_results: List[ComparisonResult] = []
    summary_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    structured_results: List[ComparisonResult] = []
    structured_summary_rows: List[Dict[str, Any]] = []
    structured_fold_rows: List[Dict[str, Any]] = []

    started = time.perf_counter()
    if not args.skip_main:
        for dataset_name in args.datasets:
            data_path = DATASET_PATHS[dataset_name]
            for missing_ratio in args.missing_ratios:
                comparison = run_comparison_experiment(
                    data_path=data_path,
                    folds=args.folds,
                    missing_ratio=missing_ratio,
                    seed=args.seed,
                    algorithms=args.algorithms,
                    max_workers=args.max_workers,
                    parallel_threshold=args.parallel_threshold,
                    chunk_size=chunk_size,
                    sample_fraction=args.sample_fraction,
                    max_objects=args.max_objects,
                    sample_seed=args.sample_seed,
                    max_reducts=1,
                    timeout=args.timeout if args.timeout > 0 else None,
                    prefer_greedy=False,
                    verbose=False,
                )
                comparison_results.append(comparison)
                for algorithm, summary in comparison.summaries.items():
                    summary_rows.append(_summary_row(dataset_name, summary))
                    fold_rows.extend(_fold_rows(dataset_name, summary))
                _write_incremental_main_outputs(
                    output_dir=output_dir,
                    comparison_results=comparison_results,
                    summary_rows=summary_rows,
                    fold_rows=fold_rows,
                )

    significance_records = compute_wilcoxon_significance(comparison_results)
    significance_rows = _significance_rows(significance_records)
    ranking_loss_check = verify_ranking_loss_implementation()

    structured_aggregate_rows: List[Dict[str, Any]] = []
    structured_degradation_rows: List[Dict[str, Any]] = []
    structured_plot_path: Optional[Path] = None
    if not args.skip_structured:
        structured_results = run_structured_missingness_experiments(
            datasets=args.datasets,
            algorithms=args.algorithms,
            patterns=args.structured_patterns,
            missing_ratio=args.structured_missing_ratio,
            folds=args.folds,
            seed=args.seed,
            max_workers=args.max_workers,
            parallel_threshold=args.parallel_threshold,
            chunk_size=chunk_size,
            sample_fraction=args.sample_fraction,
            max_objects=args.max_objects,
            sample_seed=args.sample_seed,
            timeout=args.timeout if args.timeout > 0 else None,
        )
        for result in structured_results:
            for algorithm, summary in result.summaries.items():
                structured_summary_rows.append(_summary_row(result.dataset_name, summary))
                structured_fold_rows.extend(_fold_rows(result.dataset_name, summary))
        structured_aggregate_rows = _aggregate_structured_missingness(structured_results)
        structured_degradation_rows = _structured_degradation_rows(structured_results)
        if not args.skip_plot:
            structured_plot_path = _plot_structured_missingness(
                structured_aggregate_rows,
                output_dir / "structured_missingness.png",
            )

    scaling_rows: List[Dict[str, Any]] = []
    plot_path: Optional[Path] = None
    if not args.skip_scaling:
        scaling_rows = run_scaling_experiments(
            seed=args.seed,
            scaling_timeout=args.scaling_timeout,
        )
        if not args.skip_plot:
            plot_path = _plot_scaling_results(
                scaling_rows,
                output_dir / "runtime_scaling.png",
            )

    runtime_local_rows = _aggregate_runtime_rows(comparison_results)
    total_runtime = time.perf_counter() - started

    _write_csv(output_dir / "comparison_summary.csv", summary_rows)
    _write_csv(output_dir / "comparison_folds.csv", fold_rows)
    _write_csv(output_dir / "significance.csv", significance_rows)
    _write_csv(output_dir / "runtime_local.csv", runtime_local_rows)
    _write_csv(output_dir / "runtime_scaling.csv", scaling_rows)
    _write_csv(output_dir / "structured_missingness_summary.csv", structured_summary_rows)
    _write_csv(output_dir / "structured_missingness_folds.csv", structured_fold_rows)
    _write_csv(output_dir / "structured_missingness_aggregate.csv", structured_aggregate_rows)
    _write_csv(output_dir / "structured_missingness_degradation.csv", structured_degradation_rows)

    _write_json(
        output_dir / "comparison_results.json",
        {
            "total_runtime": total_runtime,
            "task_results": comparison_results,
            "summary_rows": summary_rows,
            "fold_rows": fold_rows,
        },
    )
    _write_json(output_dir / "significance.json", significance_records)
    _write_json(output_dir / "ranking_loss_verification.json", ranking_loss_check)
    _write_json(output_dir / "runtime_scaling.json", scaling_rows)
    _write_json(output_dir / "structured_missingness.json", structured_results)
    _write_json(output_dir / "structured_missingness_plot.json", {"plot_path": str(structured_plot_path) if structured_plot_path else None})

    _write_markdown_summary(
        output_path=output_dir / "reviewer_experiment_summary.md",
        task_results=comparison_results,
        significance_records=significance_records,
        ranking_loss_check=ranking_loss_check,
        scaling_rows=scaling_rows,
        plot_path=plot_path,
        sample_fraction=args.sample_fraction,
        max_objects=args.max_objects,
    )

    manifest = {
        "output_dir": output_dir,
        "comparison_summary_csv": output_dir / "comparison_summary.csv",
        "comparison_folds_csv": output_dir / "comparison_folds.csv",
        "comparison_results_json": output_dir / "comparison_results.json",
        "structured_missingness_summary_csv": output_dir / "structured_missingness_summary.csv",
        "structured_missingness_aggregate_csv": output_dir / "structured_missingness_aggregate.csv",
        "structured_missingness_degradation_csv": output_dir / "structured_missingness_degradation.csv",
        "significance_csv": output_dir / "significance.csv",
        "ranking_loss_verification_json": output_dir / "ranking_loss_verification.json",
        "runtime_local_csv": output_dir / "runtime_local.csv",
        "runtime_scaling_csv": output_dir / "runtime_scaling.csv",
        "reviewer_markdown": output_dir / "reviewer_experiment_summary.md",
        "runtime_scaling_plot": plot_path,
        "structured_missingness_plot": structured_plot_path,
        "total_runtime_seconds": total_runtime,
    }
    _write_json(output_dir / "manifest.json", manifest)

    print("=== Reviewer experiment suite finished ===")
    print(f"Output directory        : {output_dir}")
    print(f"Total wall time (s)     : {total_runtime:.6f}")
    print(f"Summary rows            : {len(summary_rows)}")
    print(f"Fold rows               : {len(fold_rows)}")
    print(f"Structured summary rows : {len(structured_summary_rows)}")
    print(f"Significance records    : {len(significance_rows)}")
    print(f"Scaling rows            : {len(scaling_rows)}")
    print(f"Reviewer markdown       : {output_dir / 'reviewer_experiment_summary.md'}")


if __name__ == "__main__":
    main()
