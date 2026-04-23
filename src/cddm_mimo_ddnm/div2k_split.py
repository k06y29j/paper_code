"""DIV2K 标准划分：训练 800（0001–0800）、验证 100（0801–0900）、测试 100（0901–1000）。

训练集与验证集通常位于 ``DIV2K_train_HR`` / ``DIV2K_valid_HR``；测试集因发布版本不同，
可能在 ``DIV2K_test_HR`` 等目录中，按文件名数字编号收集。
"""

from __future__ import annotations

import glob
import os
import re
from typing import Iterable


_DIV2K_NAME_RE = re.compile(r"^0*(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def div2k_index_from_filename(basename: str) -> int | None:
    m = _DIV2K_NAME_RE.match(basename)
    if not m:
        return None
    return int(m.group(1))


def _indexed_paths_in_dir(directory: str) -> dict[int, str]:
    if not os.path.isdir(directory):
        return {}
    out: dict[int, str] = {}
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        for p in glob.glob(os.path.join(directory, ext)):
            idx = div2k_index_from_filename(os.path.basename(p))
            if idx is None:
                continue
            out[idx] = os.path.abspath(p)
    return out


def _range_list(d: dict[int, str], lo: int, hi: int) -> list[str]:
    missing = [i for i in range(lo, hi + 1) if i not in d]
    if missing:
        sample = missing[:10]
        more = f" ... (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"DIV2K 划分不完整：需要编号 {lo}–{hi}，缺失 {len(missing)} 个，例如 {sample}{more}"
        )
    return [d[i] for i in range(lo, hi + 1)]


def _merge_test_indices(data_dir: str, test_subdirs: Iterable[str]) -> dict[int, str]:
    merged: dict[int, str] = {}
    for sub in test_subdirs:
        path = os.path.join(data_dir, sub)
        part = _indexed_paths_in_dir(path)
        for k, v in part.items():
            if 901 <= k <= 1000:
                merged[k] = v
    return merged


def resolve_div2k_standard_splits(
    data_dir: str,
    *,
    require_test: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """返回 (train_800, valid_100, test_100) 的绝对路径列表（按编号排序）。"""
    data_dir = os.path.abspath(data_dir)
    train_map = _indexed_paths_in_dir(os.path.join(data_dir, "DIV2K_train_HR"))
    valid_map = _indexed_paths_in_dir(os.path.join(data_dir, "DIV2K_valid_HR"))

    train_paths = _range_list(train_map, 1, 800)
    valid_paths = _range_list(valid_map, 801, 900)

    test_subdirs = (
        "DIV2K_test_HR",
        "DIV2K_benchmark_HR",
        "DIV2K_test",
        "benchmark",
    )
    test_map = _merge_test_indices(data_dir, test_subdirs)
    if require_test:
        if len(test_map) < 100:
            raise FileNotFoundError(
                f"未找到完整的 100 张测试图（0901–1000）。"
                f"已在 {data_dir} 下子目录 {list(test_subdirs)} 中搜索，目前仅匹配 {len(test_map)} 张。"
                f"请将测试集 HR 放入例如 DIV2K_test_HR，或设置 require_test=False。"
            )
        test_paths = _range_list(test_map, 901, 1000)
    else:
        test_paths = [test_map[i] for i in range(901, 1001) if i in test_map]

    return train_paths, valid_paths, test_paths


def all_div2k_hr_paths_ordered(data_dir: str, *, require_test: bool = True) -> list[str]:
    """按编号 0001–1000 顺序返回 **全部 1000 张** HR 路径（train+valid+test 拼接）。"""
    tr, va, te = resolve_div2k_standard_splits(data_dir, require_test=require_test)
    all_p = tr + va + te
    if len(all_p) != 1000:
        raise FileNotFoundError(
            f"「全图 LMDB」需要恰好 1000 张 DIV2K HR（0001–1000），当前解析到 {len(all_p)} 张。"
            f"请补齐 DIV2K_train_HR / DIV2K_valid_HR / DIV2K_test_HR（0901–1000），"
            f"或勿使用 full1000 格式。"
        )
    return all_p


def write_path_manifest(paths: list[str], out_file: str) -> None:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")


def div2k_lmdb_filename(crop_size: int, num_samples: int) -> str:
    """单 patch / 旧版：每条 LMDB 记录 1 个 patch。"""
    return f"train_{crop_size}_n{num_samples}.lmdb"


def div2k_lmdb_filename_full1000(crop_size: int, num_samples: int) -> str:
    """每条 LMDB 记录含对 1000 张原图各裁 1 个 patch（共 1000×patch²）。"""
    return f"train_{crop_size}_n{num_samples}_x1000.lmdb"


def default_processed_lmdb_path(output_root: str, crop_size: int, num_samples: int) -> str:
    """处理后的 LMDB 默认路径：``<output_root>/lmdb/<name>.lmdb``。"""
    return os.path.join(output_root, "lmdb", div2k_lmdb_filename(crop_size, num_samples))


def default_processed_lmdb_path_full1000(output_root: str, crop_size: int, num_samples: int) -> str:
    return os.path.join(output_root, "lmdb", div2k_lmdb_filename_full1000(crop_size, num_samples))
