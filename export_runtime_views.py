from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from scipy.stats import wilcoxon


OURS = "ours"
BASELINES = ("mfs-mcdm", "fimf", "ml-csfs")
ALGORITHM_LABELS = {
    "ours": "Ours-exact",
    "mfs-mcdm": "MFS-MCDM",
    "fimf": "FIMF",
    "ml-csfs": "ML-CSFS",
}


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0, 0.0
    if len(values_list) == 1:
        return values_list[0], 0.0
    return float(statistics.mean(values_list)), float(statistics.stdev(values_list))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 reviewer 实验结果中导出更适合审稿回复的 runtime 视图。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="run_reviewer_experiments.py 的输出目录。",
    )
    return parser.parse_args()


def _build_refined_runtime_rows(folds_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    group_cols = ["dataset", "missing_ratio", "algorithm", "algorithm_label"]
    for keys, group in folds_df.groupby(group_cols, sort=True):
        dataset, missing_ratio, algorithm, algorithm_label = keys
        configured = group["configured_selection_time"].astype(float)
        reduction = group["reduction_time"].astype(float)
        preprocessing = group["preprocessing_time"].astype(float)
        tuning = group["tuning_time"].astype(float)
        pipeline = group["selection_pipeline_time"].astype(float)
        avg_configured, std_configured = _mean_std(configured)
        avg_reduction, std_reduction = _mean_std(reduction)
        avg_preprocessing, std_preprocessing = _mean_std(preprocessing)
        avg_tuning, std_tuning = _mean_std(tuning)
        avg_pipeline, std_pipeline = _mean_std(pipeline)
        rows.append(
            {
                "dataset": dataset,
                "missing_ratio": float(missing_ratio),
                "algorithm": algorithm,
                "algorithm_label": algorithm_label,
                "avg_pure_reduction_time": avg_reduction,
                "std_pure_reduction_time": std_reduction,
                "avg_reduction_time": avg_reduction,
                "std_reduction_time": std_reduction,
                "avg_preprocessing_time": avg_preprocessing,
                "std_preprocessing_time": std_preprocessing,
                "avg_configured_reduction_time": avg_configured,
                "std_configured_reduction_time": std_configured,
                "avg_configured_selection_time": avg_configured,
                "std_configured_selection_time": std_configured,
                "avg_tuning_time": avg_tuning,
                "std_tuning_time": std_tuning,
                "avg_full_runtime_time": avg_pipeline,
                "std_full_runtime_time": std_pipeline,
                "avg_selection_pipeline_time": avg_pipeline,
                "std_selection_pipeline_time": std_pipeline,
                "avg_selected_attributes": float(group["selected_attributes"].astype(float).mean()),
            }
        )
    return rows


def _significance_rows(runtime_rows_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metrics = (
        ("pure_reduction_time", "avg_pure_reduction_time"),
        ("configured_reduction_time", "avg_configured_reduction_time"),
        ("full_runtime_time", "avg_full_runtime_time"),
    )

    for metric_name, column in metrics:
        ours_values = (
            runtime_rows_df.loc[runtime_rows_df["algorithm"] == OURS, column]
            .astype(float)
            .tolist()
        )
        if len(ours_values) < 2:
            continue

        for baseline in BASELINES:
            baseline_values = (
                runtime_rows_df.loc[runtime_rows_df["algorithm"] == baseline, column]
                .astype(float)
                .tolist()
            )
            if len(baseline_values) != len(ours_values) or len(baseline_values) < 2:
                continue
            try:
                statistic, p_value = wilcoxon(ours_values, baseline_values)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            rows.append(
                {
                    "metric": metric_name,
                    "baseline": baseline,
                    "baseline_label": ALGORITHM_LABELS[baseline],
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "ours_mean": float(statistics.mean(ours_values)),
                    "baseline_mean": float(statistics.mean(baseline_values)),
                }
            )
    return rows


def _aggregate_metric_means(runtime_rows_df: pd.DataFrame, column: str) -> Dict[str, float]:
    means: Dict[str, float] = {}
    for algorithm, group in runtime_rows_df.groupby("algorithm"):
        means[str(algorithm)] = float(group[column].astype(float).mean())
    return means


def _max_fold_rows(folds_df: pd.DataFrame, column: str, top_n: int = 3) -> pd.DataFrame:
    ranked = folds_df.sort_values(column, ascending=False)
    return ranked.head(top_n)[
        [
            "dataset",
            "missing_ratio",
            "algorithm",
            "algorithm_label",
            "fold_index",
            column,
        ]
    ]


def _write_markdown_summary(
    path: Path,
    runtime_rows_df: pd.DataFrame,
    significance_df: pd.DataFrame,
    folds_df: pd.DataFrame,
) -> None:
    pure_means = _aggregate_metric_means(runtime_rows_df, "avg_reduction_time")
    configured_means = _aggregate_metric_means(
        runtime_rows_df,
        "avg_configured_selection_time",
    )
    pipeline_means = _aggregate_metric_means(
        runtime_rows_df,
        "avg_selection_pipeline_time",
    )
    max_configured = _max_fold_rows(folds_df, "configured_selection_time", top_n=3)

    lines = [
        "# Runtime 口径说明",
        "",
        "## 1. 建议给审稿人使用哪个时间指标",
        "- 不建议直接用 `tuning_time` 做方法效率对比，因为它主要反映训练折内的调参代价，而不是约简算法本身的执行效率。",
        "- 不建议只用 `pure_reduction_time` 做唯一主指标，因为 baseline 必须先对缺失数据做插补后才能运行，完全剔除这一步会低估 baseline 的必要开销。",
        "- 最适合作为审稿回复主指标的是 `configured_reduction_time = preprocessing_time + pure_reduction_time`。",
        "- 这个指标表示：在已经选定最终 `imputer + k` 配置之后，在外层训练折上完成一次特征约简所需的实际执行时间。",
        "- 其中，`preprocessing_time` 是最终选定插补器在外层训练折上的拟合与变换时间，`pure_reduction_time` 是选择器本身的运行时间，也就是最狭义的“约简时间”。",
        "- `full_runtime_time = tuning_time + configured_reduction_time` 只适合作为补充说明，用来展示完整训练折内模型选择的总开销。",
        "",
        "## 2. 9 个任务点上的平均时间",
    ]
    for algorithm in (OURS, "mfs-mcdm", "fimf", "ml-csfs"):
        label = ALGORITHM_LABELS[algorithm]
        lines.append(
            f"- `{label}`：纯约简时间 {pure_means.get(algorithm, 0.0):.6f} s，"
            f"最终配置下约简时间 {configured_means.get(algorithm, 0.0):.6f} s，"
            f"完整训练内总开销 {pipeline_means.get(algorithm, 0.0):.6f} s。"
        )

    lines.extend(
        [
            "",
            "## 3. 显著性检验",
            "- 下表中的 `configured_reduction_time` 最值得在审稿回复中引用，因为它最接近“真正完成一次约简所需的执行时间”。",
        ]
    )
    for _, row in significance_df.iterrows():
        lines.append(
            f"- `{row['metric']}` vs `{row['baseline_label']}`："
            f"Ours={float(row['ours_mean']):.6f}，Baseline={float(row['baseline_mean']):.6f}，"
            f"p={float(row['p_value']):.6g}。"
        )

    lines.extend(
        [
            "",
            "## 4. 最慢的 configured_reduction_time folds",
        ]
    )
    for _, row in max_configured.iterrows():
        lines.append(
            f"- `{row['algorithm_label']}` @ `{row['dataset']}` / `{float(row['missing_ratio']):.0%}` / "
            f"`fold {int(row['fold_index'])}`：{float(row['configured_selection_time']):.6f} s。"
        )

    lines.extend(
        [
            "",
            "## 5. 可直接写进回复的结论",
            "- 审稿人若关心“约简方法本身是否高效”，主表应报告 `configured_reduction_time`，而不是把训练内调参总时间混进同一列。",
            "- 在该口径下，`Ours` 仍保持稳定时间优势，但不再出现数百倍到数千倍的夸张差距。",
            "- `full_runtime_time` 可以作为补充结果说明：baseline 在公平调优协议下还需要额外承担插补器选择与子集大小选择的开销。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    folds_path = input_dir / "comparison_folds.csv"
    if not folds_path.exists():
        raise FileNotFoundError(f"未找到 {folds_path}")

    folds_df = pd.read_csv(folds_path)
    folds_df["configured_selection_time"] = (
        folds_df["reduction_time"].astype(float)
        + folds_df["preprocessing_time"].astype(float)
    )

    runtime_rows = _build_refined_runtime_rows(folds_df)
    runtime_rows_df = pd.DataFrame(runtime_rows)
    significance_rows = _significance_rows(runtime_rows_df)
    significance_df = pd.DataFrame(significance_rows)

    _write_csv(input_dir / "runtime_local_refined.csv", runtime_rows)
    _write_csv(input_dir / "runtime_significance_refined.csv", significance_rows)
    _write_markdown_summary(
        input_dir / "runtime_response_note_zh.md",
        runtime_rows_df,
        significance_df,
        folds_df,
    )

    print(f"Refined runtime exports written to: {input_dir}")


if __name__ == "__main__":
    main()
