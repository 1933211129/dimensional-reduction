from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from algorithm1 import MISSING_VALUE


def parse_attribute_line(line: str) -> Tuple[str, str]:
    """解析 ARFF 文件中的 @attribute 行。"""
    stripped = line.strip()
    if not stripped.lower().startswith("@attribute"):
        raise ValueError(f"Invalid attribute declaration: {stripped}")
    content = stripped[len("@attribute") :].strip()
    if not content:
        raise ValueError(f"Invalid attribute declaration: {stripped}")

    if content[0] in {"'", '"'}:
        quote = content[0]
        idx = 1
        name_chars: List[str] = []
        escaped = False
        while idx < len(content):
            ch = content[idx]
            if escaped:
                name_chars.append(ch)
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                break
            else:
                name_chars.append(ch)
            idx += 1
        else:
            raise ValueError(f"Unterminated quoted attribute name: {stripped}")
        name = "".join(name_chars)
        idx += 1
        type_part = content[idx:].strip()
    else:
        parts = content.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid attribute declaration: {stripped}")
        name, type_part = parts[0], parts[1].strip()

    if not type_part:
        raise ValueError(f"Missing attribute type for {name!r}")

    return name, type_part


_RELATION_LABEL_COUNT_PATTERN = re.compile(r"-c\s+-?(\d+)")


def extract_label_count(relation_line: str) -> Optional[int]:
    """从 @relation 行中提取标签数量提示。"""
    match = _RELATION_LABEL_COUNT_PATTERN.search(relation_line)
    if match:
        return int(match.group(1))
    return None


def normalize_type(type_part: str) -> str:
    """规范化类型字符串。"""
    return type_part.strip().lower()


def is_numeric_type(type_part: str) -> bool:
    """判断是否为数值类型。"""
    return normalize_type(type_part) in {"numeric", "real", "integer"}


def is_binary_label_type(type_part: str) -> bool:
    """判断是否为二元标签类型。"""
    normalized = type_part.strip().lower().replace(" ", "")
    return normalized in {"{0,1}", "{1,0}"}


def split_nominal_values(type_part: str) -> List[str]:
    """解析名义值集合，返回值列表。"""
    inner = type_part.strip()
    if not inner.startswith("{") or not inner.endswith("}"):
        return []
    inner = inner[1:-1]
    if not inner:
        return []
    reader = csv.reader(
        [inner],
        delimiter=",",
        quotechar="'",
        escapechar="\\",
        skipinitialspace=True,
    )
    raw_values = next(reader, [])
    cleaned: List[str] = []
    for raw in raw_values:
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        cleaned.append(value)
    return cleaned


def encode_nominal_value(
    attr_name: str,
    type_part: str,
    raw_value: str,
    *,
    cache: Dict[str, Dict[str, float]],
) -> float:
    """将名义值编码为浮点数。"""
    mapping = cache.get(attr_name)
    if mapping is None:
        mapping = {}
        options = split_nominal_values(type_part)
        for option in options:
            if option in mapping:
                continue
            try:
                mapping[option] = float(option)
            except ValueError:
                mapping[option] = float(len(mapping))
        cache[attr_name] = mapping

    value_key = raw_value.strip()
    if len(value_key) >= 2 and value_key[0] == value_key[-1] and value_key[0] in {"'", '"'}:
        value_key = value_key[1:-1]

    if value_key in mapping:
        return mapping[value_key]

    try:
        return float(value_key)
    except ValueError as exc:
        raise ValueError(
            f"值 {raw_value!r} 不在属性 {attr_name!r} 的枚举集合 {list(mapping.keys())!r} 中。"
        ) from exc


def split_attribute_specs(
    specs: List[Tuple[str, str]],
    label_count_hint: Optional[int],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """将属性规格列表分割为特征和标签。"""
    if not specs:
        raise ValueError("ARFF 文件缺少属性定义。")

    if label_count_hint is not None:
        if label_count_hint < 0 or label_count_hint > len(specs):
            raise ValueError(
                f"Relation 中的标签数量提示 {label_count_hint} 与属性数不匹配（总计 {len(specs)}）。"
            )
        split_index = len(specs) - label_count_hint
        feature_specs = specs[:split_index] if split_index > 0 else []
        label_specs = specs[split_index:]
    else:
        label_specs: List[Tuple[str, str]] = []
        for name, type_part in reversed(specs):
            if is_binary_label_type(type_part):
                label_specs.append((name, type_part))
            else:
                break
        label_specs.reverse()
        feature_specs = specs[: len(specs) - len(label_specs)]

    if not label_specs:
        raise ValueError("无法识别标签属性，请确认 ARFF 文件的 Relation 信息或标签定义。")

    for name, type_part in label_specs:
        if not is_binary_label_type(type_part):
            raise ValueError(f"标签 {name!r} 类型不是二元标签：{type_part}")

    return feature_specs, label_specs


def read_arff(path: Path) -> Tuple[List[str], List[List[float]], List[str], List[List[int]]]:
    """读取 ARFF 文件，返回特征名称、特征行、标签名称、标签行。"""
    feature_names: List[str] = []
    label_names: List[str] = []
    feature_rows: List[List[float]] = []
    label_rows: List[List[int]] = []
    attribute_specs: List[Tuple[str, str]] = []
    feature_specs: List[Tuple[str, str]] = []
    label_specs: List[Tuple[str, str]] = []
    nominal_cache: Dict[str, Dict[str, float]] = {}
    relation_label_count: Optional[int] = None
    in_data = False

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lowered = line.lower()

            if lowered.startswith("@relation"):
                relation_label_count = extract_label_count(lowered)
                continue

            if lowered.startswith("@attribute"):
                name, type_part = parse_attribute_line(line)
                attribute_specs.append((name, type_part))
                continue

            if lowered.startswith("@data"):
                in_data = True
                feature_specs, label_specs = split_attribute_specs(
                    attribute_specs,
                    relation_label_count,
                )
                feature_names = [name for name, _ in feature_specs]
                label_names = [name for name, _ in label_specs]
                continue

            if not in_data:
                continue

            if not feature_specs or not label_specs:
                raise ValueError("在读取数据前尚未解析出属性定义。")

            values = [value.strip() for value in line.split(",")]
            expected_len = len(feature_specs) + len(label_specs)
            if len(values) != expected_len:
                raise ValueError(
                    f"Expected {expected_len} values per row, got {len(values)}: {line}"
                )

            feature_values: List[float] = []
            for (name, type_part), raw_value in zip(feature_specs, values[: len(feature_specs)]):
                if raw_value == "?":
                    feature_values.append(float("nan"))
                    continue

                if is_numeric_type(type_part):
                    try:
                        feature_values.append(float(raw_value))
                    except ValueError as exc:
                        raise ValueError(
                            f"属性 {name!r} 期望数值类型，但遇到无法解析的值：{raw_value!r}"
                        ) from exc
                    continue

                if type_part.strip().lower() == "string" or type_part.strip().startswith("{"):
                    feature_values.append(
                        encode_nominal_value(name, type_part, raw_value, cache=nominal_cache)
                    )
                    continue

                raise ValueError(f"Unsupported attribute type for feature {name!r}: {type_part}")

            label_values: List[int] = []
            for (name, type_part), raw_value in zip(
                label_specs, values[len(feature_specs) : len(feature_specs) + len(label_specs)]
            ):
                if raw_value == "?":
                    raise ValueError(f"标签 {name!r} 存在缺失值，当前实现不支持。")
                try:
                    numeric_value = float(raw_value)
                except ValueError as exc:
                    normalized = raw_value.strip().lower()
                    if normalized in {"yes", "true"}:
                        numeric_value = 1.0
                    elif normalized in {"no", "false"}:
                        numeric_value = 0.0
                    else:
                        raise ValueError(
                            f"标签 {name!r} 包含无法解析的取值：{raw_value!r}"
                        ) from exc
                int_value = int(round(numeric_value))
                if int_value not in (0, 1):
                    raise ValueError(f"标签 {name!r} 不是二元值：{raw_value!r}")
                label_values.append(int_value)

            feature_rows.append(feature_values)
            label_rows.append(label_values)

    return feature_names, feature_rows, label_names, label_rows

