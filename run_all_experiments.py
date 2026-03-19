from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from experiment_core import (
    DATA_DIR,
    DEFAULT_FOLDS,
    MISSING_RATIO,
    RANDOM_SEED,
    run_experiment,
)


DATASET_CONFIGS = {
    "yeast": ("Yeast Multi-Label Reduction Experiment (K-fold)", DATA_DIR / "yeast.arff"),
    "birds": ("Birds Multi-Label Reduction Experiment (K-fold)", DATA_DIR / "birds.arff"),
    "scene": ("Scene Multi-Label Reduction Experiment (K-fold)", DATA_DIR / "scene.arff"),
}

DATASET_DEFAULT_PARAMS: Dict[str, Dict[str, object]] = {
    "birds": {
        "prefer_greedy": False,
        "max_reducts": 1,
        "timeout": 120.0,
    }
}

DEFAULT_MISSING_RATIOS = (MISSING_RATIO, 0.10, 0.15)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="顺序运行多个多标签约简实验，支持对多组缺失率逐一执行，并将结果打印到同一日志。",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIGS.keys()),
        default=list(DATASET_CONFIGS.keys()),
        help="要运行的数据集标识，默认依次运行全部（yeast、birds、scene）。",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_FOLDS,
        help="交叉验证折数，默认值与单数据集脚本一致。",
    )
    parser.add_argument(
        "--missing-ratios",
        nargs="+",
        type=float,
        default=None,
        metavar="RATIO",
        help="注入缺失值比例列表，缺省时为脚本内置的三组比例。",
    )
    parser.add_argument(
        "--missing-ratio",
        type=float,
        default=None,
        help="兼容旧参数：指定单一缺失值比例。若同时提供 --missing-ratios，将以新参数为准。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="随机种子，用于折分和缺失值注入。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="容差计算最大并行进程数，缺省时沿用系统 CPU 数。",
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
    return parser.parse_args()


def _validate_paths(dataset_keys: List[str]) -> None:
    missing = []
    for key in dataset_keys:
        title, path = DATASET_CONFIGS[key]
        if not Path(path).is_file():
            missing.append(f"{key} -> {path}")
    if missing:
        raise FileNotFoundError(
            "以下数据集文件不存在，请检查路径：\n" + "\n".join(missing)
        )


def main() -> None:
    args = _parse_args()
    dataset_keys: List[str] = args.datasets
    _validate_paths(dataset_keys)

    if args.missing_ratios is not None and args.missing_ratio is not None:
        print("[warn] 同时指定了 --missing-ratios 与 --missing-ratio，已忽略旧参数。")

    if args.missing_ratios is not None:
        missing_ratios = args.missing_ratios
    elif args.missing_ratio is not None:
        missing_ratios = [args.missing_ratio]
    else:
        seen = set()
        missing_ratios = []
        for ratio in DEFAULT_MISSING_RATIOS:
            if ratio not in seen:
                seen.add(ratio)
                missing_ratios.append(ratio)

    chunk_size = args.chunk_size if args.chunk_size > 0 else None

    for ratio_index, missing_ratio in enumerate(missing_ratios, start=1):
        ratio_label = f"{missing_ratio:.2%}"
        if ratio_index > 1:
            print("\n" + "#" * 80 + "\n")
        print(f"### Missing ratio: {ratio_label}")

        for dataset_index, key in enumerate(dataset_keys, start=1):
            title, path = DATASET_CONFIGS[key]
            if dataset_index > 1:
                print("\n" + "=" * 80 + "\n")
            extra_params = DATASET_DEFAULT_PARAMS.get(key, {})
            run_experiment(
                data_path=Path(path),
                folds=args.folds,
                missing_ratio=missing_ratio,
                seed=args.seed,
                max_workers=args.max_workers,
                parallel_threshold=args.parallel_threshold,
                chunk_size=chunk_size,
                title=f"{title} | Missing {ratio_label}",
                **extra_params,
            )


if __name__ == "__main__":
    main()
