from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from experiment_core import (
    DATA_DIR,
    DEFAULT_FOLDS,
    DEFAULT_MISSING_RATIOS,
    RANDOM_SEED,
    run_experiment,
)


DATA_PATH = DATA_DIR / "scene.arff"
DEFAULT_TITLE = "Scene Multi-Label Reduction Experiment (K-fold)"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scene 多标记约简实验（支持并行和 K 折交叉验证）。"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="ARFF 数据集路径，默认为项目自带的 scene 数据。",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_FOLDS,
        help="交叉验证折数，默认 10。",
    )
    parser.add_argument(
        "--missing-ratio",
        type=float,
        default=None,
        help="注入缺失值比例。与 --missing-ratios 同时使用时将被忽略。",
    )
    parser.add_argument(
        "--missing-ratios",
        nargs="+",
        type=float,
        default=None,
        metavar="RATIO",
        help="注入缺失值比例列表，默认依次为 0.05 0.10 0.15。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="随机种子，用于缺失值注入与折分。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="容差计算使用的最大并行进程数，缺省时等于 CPU 核心数。",
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
        help="ProcessPoolExecutor map 的 chunksize；小于等于 0 时按默认策略估计。",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="对样本行进行随机抽样的比例 (0, 1]，默认不抽样。",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="限制用于实验的最大样本行数，默认不限。",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="抽样时使用的随机种子，默认沿用 --seed。",
    )
    return parser.parse_args()


def _resolve_missing_ratios(args: argparse.Namespace) -> List[float]:
    if args.missing_ratios is not None and args.missing_ratio is not None:
        print("[warn] 同时指定了 --missing-ratios 与 --missing-ratio，已忽略单一比例参数。")

    if args.missing_ratios is not None:
        return list(args.missing_ratios)

    if args.missing_ratio is not None:
        return [args.missing_ratio]

    seen = set()
    ratios: List[float] = []
    for ratio in DEFAULT_MISSING_RATIOS:
        if ratio not in seen:
            seen.add(ratio)
            ratios.append(ratio)
    return ratios


def main() -> None:
    args = _parse_args()
    chunk_size = args.chunk_size if args.chunk_size > 0 else None
    missing_ratios = _resolve_missing_ratios(args)

    for ratio_index, missing_ratio in enumerate(missing_ratios, start=1):
        ratio_label = f"{missing_ratio:.2%}"
        if ratio_index > 1:
            print("\n" + "#" * 80 + "\n")
        run_experiment(
            data_path=args.data_path,
            folds=args.folds,
            missing_ratio=missing_ratio,
            seed=args.seed,
            max_workers=args.max_workers,
            parallel_threshold=args.parallel_threshold,
            chunk_size=chunk_size,
            title=f"{DEFAULT_TITLE} | Missing {ratio_label}",
            sample_fraction=args.sample_fraction,
            max_objects=args.max_objects,
            sample_seed=args.sample_seed,
        )


if __name__ == "__main__":
    main()


