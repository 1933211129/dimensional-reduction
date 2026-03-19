from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType


BASE_DIR = Path(__file__).resolve().parent
ALGORITHMS_DIR = BASE_DIR / "algorithms"


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为模块 {module_name!r} 创建加载规范：{file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_namespace_package(package_name: str, package_dir: Path) -> ModuleType:
    existing = sys.modules.get(package_name)
    if existing is not None:
        return existing

    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    sys.modules[package_name] = package
    return package


def load_fimf_module() -> ModuleType:
    return _load_module("codex_fimf", ALGORITHMS_DIR / "fimf.py")


def load_mfs_mcdm_module() -> ModuleType:
    return _load_module("codex_mfs_mcdm", ALGORITHMS_DIR / "mfs_mcdm.py")


def load_mlcsfs_feature_selection_module() -> ModuleType:
    package_dir = ALGORITHMS_DIR / "ml_csfs"
    _ensure_namespace_package("codex_mlcsfs", package_dir)
    _load_module("codex_mlcsfs.data_utils", package_dir / "data_utils.py")
    return _load_module(
        "codex_mlcsfs.feature_selection",
        package_dir / "feature_selection.py",
    )


def load_mlknn_module() -> ModuleType:
    package_dir = ALGORITHMS_DIR / "ml_csfs"
    _ensure_namespace_package("codex_mlcsfs", package_dir)
    return _load_module("codex_mlcsfs.mlknn", package_dir / "mlknn.py")
