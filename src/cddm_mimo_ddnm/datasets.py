"""数据集加载模块。

支持：
  - CIFAR-10（两种格式自动识别）：
      1. torchvision 标准格式：<data_dir>/cifar-10-batches-py/  （本地下载）
      2. HuggingFace Parquet 格式：<data_dir>/plain_text/*.parquet
         每行含 {'img': {'bytes': <PNG bytes>}, 'label': int}
         （如 /data/small-datasets-1/cifar10/plain_text/train-*.parquet）
  - DIV2K（高清图像，训练时 RandomCrop，验证时 CenterCrop）
      - 图像文件：DIV2K_train_HR/*.png, DIV2K_valid_HR/*.png
      - LMDB：DIV2K_train_HR/train-{crop_size}.lmdb, DIV2K_valid_HR/valid-{crop_size}.lmdb
        （若存在则优先使用 LMDB，支持 .keys.cache 加速）
"""

from __future__ import annotations

import glob
import io
import os
import pickle

import numpy as np
import torch
import torchvision.datasets as tvdatasets
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# ---------------------------------------------------------------------------
# LMDB 记录格式（供 build_div2k_lmdb.py 写入，本模块读取）
# ---------------------------------------------------------------------------

class LMDBImageRecord:
    """单张 patch 的 LMDB 记录（numpy [H,W,3] uint8）。"""

    def __init__(self, img: np.ndarray) -> None:
        self.img = np.asarray(img, dtype=np.uint8)


class LMDBFullDiv2KRecord:
    """full1000 格式：每条记录含多张 patch  stacked 为 [N, H, W, 3]。"""

    def __init__(self, arr: np.ndarray) -> None:
        self.arr = np.asarray(arr, dtype=np.uint8)


# ---------------------------------------------------------------------------
# DIV2K LMDB Dataset
# ---------------------------------------------------------------------------

class DIV2KLMDBDataset(Dataset):
    """DIV2K 的 LMDB 格式数据集。

    支持目录结构：
      <data_dir>/DIV2K_train_HR/train-{crop_size}.lmdb  (及可选 train-{crop_size}.lmdb.keys.cache)
      <data_dir>/DIV2K_valid_HR/valid-{crop_size}.lmdb (及可选 valid-{crop_size}.lmdb.keys.cache)

    LMDB 内每条记录支持多种格式：
      - pickle(LMDBImageRecord): 单张 patch
      - pickle(LMDBFullDiv2KRecord): 多 patch，取第一张或随机一张（训练时）
      - pickle(numpy.ndarray): [H,W,3] 或 [N,H,W,3]
      - 原始 PNG/JPEG bytes
    """

    def __init__(
        self,
        lmdb_path: str,
        crop_size: int,
        split: str,
        keys_cache_path: str | None = None,
    ) -> None:
        super().__init__()
        import lmdb

        self.lmdb_path = lmdb_path
        self.crop_size = crop_size
        self.split = split
        self._lmdb_subdir = os.path.isdir(lmdb_path)
        self._env: lmdb.Environment | None = None

        if split == "train":
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
        else:
            self.transform = T.ToTensor()

        # 加载 key 列表
        self._keys: list[bytes] = []
        if keys_cache_path and os.path.isfile(keys_cache_path):
            try:
                with open(keys_cache_path, "rb") as f:
                    raw_keys = pickle.load(f)
                self._keys = [
                    k.encode("ascii") if isinstance(k, str) else k
                    for k in (raw_keys if isinstance(raw_keys, (list, tuple)) else [])
                ]
            except Exception:
                pass

        env = lmdb.open(
            lmdb_path,
            subdir=self._lmdb_subdir,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        try:
            with env.begin(write=False) as txn:
                if not self._keys:
                    # 尝试从 LMDB 内读取 __keys__
                    meta = txn.get(b"__keys__")
                    if meta is not None:
                        raw = pickle.loads(meta)
                        self._keys = [
                            k.encode("ascii") if isinstance(k, str) else k
                            for k in (raw if isinstance(raw, (list, tuple)) else [])
                        ]
                    else:
                        self._keys = [
                            k for k, _ in txn.cursor()
                            if not k.startswith(b"__")
                        ]
        finally:
            env.close()

        if not self._keys:
            raise ValueError(
                f"DIV2K LMDB：{lmdb_path} 中未找到有效记录，"
                f"请确认 LMDB 格式或检查 .keys.cache。"
            )

        # 启动前做轻量体检：抽样验证若干条记录可被当前解码逻辑读取。
        # 若失败，交由上层回退到图像文件模式，避免训练中途才报错。
        check_keys = self._keys[: min(8, len(self._keys))]
        env_check = lmdb.open(
            lmdb_path,
            subdir=self._lmdb_subdir,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=64,
        )
        try:
            with env_check.begin(write=False) as txn:
                for k in check_keys:
                    v = txn.get(k)
                    if v is None:
                        raise ValueError(f"LMDB key 缺失：{k!r}")
                    _ = self._decode_value(v)
        except Exception as e:
            raise ValueError(f"LMDB 抽样解码失败：{e}") from e
        finally:
            env_check.close()

    def _decode_value(self, val: bytes) -> np.ndarray:
        """将 LMDB 值解码为 [H,W,3] uint8 numpy 数组。"""
        # 尝试 pickle（LMDBImageRecord / LMDBFullDiv2KRecord / 裸 numpy / dict / tuple）
        try:
            obj = pickle.loads(val)
            arr = None
            if hasattr(obj, "img"):
                arr = np.asarray(obj.img, dtype=np.uint8)
            elif hasattr(obj, "arr"):
                arr = np.asarray(obj.arr, dtype=np.uint8)
                if arr.ndim == 4:
                    i = np.random.randint(0, arr.shape[0]) if self.split == "train" else 0
                    arr = arr[i]
            elif isinstance(obj, np.ndarray):
                arr = np.asarray(obj, dtype=np.uint8)
                if arr.ndim == 4:
                    arr = arr[0]
            elif isinstance(obj, dict):
                if "data" in obj and "shape" in obj:
                    arr = np.frombuffer(obj["data"], dtype=np.uint8).reshape(obj["shape"])
                else:
                    for k in ("img", "image", "data", "arr", "patch"):
                        if k in obj:
                            arr = np.asarray(obj[k], dtype=np.uint8)
                            if arr.ndim == 4:
                                arr = arr[0]
                            break
            elif isinstance(obj, (tuple, list)) and len(obj) > 0:
                arr = np.asarray(obj[0], dtype=np.uint8)
                if arr.ndim == 4:
                    arr = arr[0]
            else:
                arr = np.asarray(obj, dtype=np.uint8)
            if arr is not None and arr.ndim >= 2:
                return arr
        except Exception:
            # 兼容历史 LMDB：pickle 中记录类定义在 __main__（如 LMDB_Image）。
            # 这类对象在当前进程找不到原类时会反序列化失败，故使用兼容 Unpickler。
            try:
                class _CompatUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        try:
                            return super().find_class(module, name)
                        except Exception:
                            # 回退为动态占位类，保留其 __dict__ 以便读取字段。
                            return type(name, (), {})

                obj = _CompatUnpickler(io.BytesIO(val)).load()
                arr = None
                if hasattr(obj, "image") and isinstance(getattr(obj, "image"), (bytes, bytearray)):
                    h, w = getattr(obj, "size", (None, None))
                    c = int(getattr(obj, "channels", 3))
                    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0 and c > 0:
                        raw = np.frombuffer(obj.image, dtype=np.uint8)
                        if raw.size == h * w * c:
                            arr = raw.reshape(h, w, c)
                if arr is None and hasattr(obj, "img"):
                    arr = np.asarray(obj.img, dtype=np.uint8)
                if arr is None and hasattr(obj, "arr"):
                    arr = np.asarray(obj.arr, dtype=np.uint8)
                    if arr.ndim == 4:
                        i = np.random.randint(0, arr.shape[0]) if self.split == "train" else 0
                        arr = arr[i]
                if arr is None and hasattr(obj, "__dict__"):
                    d = obj.__dict__
                    if isinstance(d, dict):
                        if "data" in d and "shape" in d:
                            arr = np.frombuffer(d["data"], dtype=np.uint8).reshape(d["shape"])
                        elif "image" in d and isinstance(d["image"], (bytes, bytearray)) and "size" in d:
                            h, w = d["size"]
                            c = int(d.get("channels", 3))
                            raw = np.frombuffer(d["image"], dtype=np.uint8)
                            if raw.size == h * w * c:
                                arr = raw.reshape(h, w, c)
                if arr is not None:
                    arr = np.asarray(arr, dtype=np.uint8)
                    if arr.ndim == 4:
                        arr = arr[0]
                    if arr.ndim >= 2:
                        return arr
            except Exception:
                pass

        # 原始图像字节（PNG/JPEG）或 OpenCV 编码
        try:
            img = Image.open(io.BytesIO(val)).convert("RGB")
            return np.array(img)
        except Exception:
            pass
        try:
            import cv2
            buf = np.frombuffer(val, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        raise ValueError(
            "LMDB 记录解码失败：既非 pickle(numpy/dict/record)，也非 PNG/JPEG/OpenCV 编码。"
            "若使用自定义 LMDB，请确保每条为 pickle(numpy [H,W,3]) 或含 img/arr 字段。"
        )

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import lmdb

        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                subdir=self._lmdb_subdir,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )

        key = self._keys[idx]
        with self._env.begin(write=False) as txn:
            val = txn.get(key)
        if val is None:
            raise KeyError(f"LMDB key {key} 不存在")

        arr = self._decode_value(val)
        # 若已裁好且尺寸匹配，直接使用；否则做 CenterCrop（验证）或 RandomCrop（训练）
        h, w = arr.shape[:2]
        if h >= self.crop_size and w >= self.crop_size:
            if self.split == "train":
                top = np.random.randint(0, h - self.crop_size + 1)
                left = np.random.randint(0, w - self.crop_size + 1)
            else:
                top = (h - self.crop_size) // 2
                left = (w - self.crop_size) // 2
            arr = arr[top : top + self.crop_size, left : left + self.crop_size]
        else:
            img = Image.fromarray(arr)
            if self.split == "train":
                arr = np.array(T.RandomCrop(self.crop_size)(img))
            else:
                arr = np.array(T.CenterCrop(self.crop_size)(img))

        img = Image.fromarray(arr)
        return self.transform(img)


# ---------------------------------------------------------------------------
# DIV2K 自定义 Dataset（图像文件）
# ---------------------------------------------------------------------------

class DIV2KDataset(Dataset):
    """DIV2K 高分辨率图像数据集（随机裁剪训练 / 中心裁剪验证）。

    Args:
        data_dir:   数据集根目录（含 DIV2K_train_HR / DIV2K_valid_HR 子目录，
                    或直接包含图像文件）。
        crop_size:  裁剪尺寸，应为 32 的整数倍（满足 patch_size=4 且 U-Net 4× 下采样）。
        split:      "train" 或 "valid"。
    """

    def __init__(self, data_dir: str, crop_size: int = 256, split: str = "train") -> None:
        super().__init__()
        hr_dir = os.path.join(data_dir, f"DIV2K_{split}_HR")
        if not os.path.isdir(hr_dir):
            hr_dir = data_dir

        paths: list[str] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
            paths.extend(glob.glob(os.path.join(hr_dir, ext)))
        self.image_paths = sorted(set(paths))

        if not self.image_paths:
            raise FileNotFoundError(
                f"DIV2K：在 {hr_dir} 下未找到任何图像文件，"
                f"请确认路径包含 DIV2K_train_HR / DIV2K_valid_HR 子目录或直接含图像。"
            )

        if split == "train":
            self.transform = T.Compose([
                T.RandomCrop(crop_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
        else:
            self.transform = T.Compose([
                T.CenterCrop(crop_size),
                T.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def _make_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int,
    num_workers: int,
    distributed: bool,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    """内部：构造 train/val DataLoader，分布式时自动添加 DistributedSampler。"""
    sampler: DistributedSampler | None = None
    if distributed:
        sampler = DistributedSampler(train_ds, shuffle=True)

    # prefetch_factor：多 worker 时加大预取，减轻 GPU 等 LMDB/解码的等待（利用率低时常有效）
    train_kw: dict = {
        "batch_size": batch_size,
        "shuffle": (sampler is None),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": (num_workers > 0),
    }
    if num_workers > 0:
        train_kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, **train_kw)

    val_kw: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": (num_workers > 0),
    }
    if num_workers > 0:
        val_kw["prefetch_factor"] = 4
    val_loader = DataLoader(val_ds, **val_kw)
    return train_loader, val_loader, sampler


# ---------------------------------------------------------------------------
# CIFAR-10：HuggingFace Parquet 格式 Dataset
# ---------------------------------------------------------------------------

class CIFAR10ParquetDataset(Dataset):
    """从 HuggingFace Parquet 文件加载 CIFAR-10。

    兼容目录结构：
      <data_dir>/plain_text/train-*.parquet  （训练集）
      <data_dir>/plain_text/test-*.parquet   （测试集）

    每行格式：{'img': {'bytes': <PNG bytes>, 'path': ...}, 'label': int}
    返回 (image_tensor, label)，与 torchvision.datasets.CIFAR10 接口一致。
    """

    def __init__(self, data_dir: str, split: str = "train", transform=None) -> None:
        super().__init__()
        import pandas as pd

        pattern_key = "train" if split == "train" else "test"
        parquet_dir = os.path.join(data_dir, "plain_text")
        files = sorted(glob.glob(os.path.join(parquet_dir, f"{pattern_key}-*.parquet")))
        if not files:
            raise FileNotFoundError(
                f"CIFAR-10 Parquet：在 {parquet_dir} 下未找到 '{pattern_key}-*.parquet' 文件，"
                f"请确认数据路径。"
            )
        self.df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_bytes = row["img"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        label = int(row["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# CIFAR-10 DataLoader 工厂（自动识别数据格式）
# ---------------------------------------------------------------------------

def get_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    distributed: bool = False,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    """CIFAR-10 DataLoader，自动识别两种数据格式。

    格式1（torchvision 标准）：<data_dir>/cifar-10-batches-py/
    格式2（HuggingFace Parquet）：<data_dir>/plain_text/train-*.parquet

    两种格式均返回 (image_tensor [C,H,W], label) 的 DataLoader。
    """
    tv_format_dir = os.path.join(data_dir, "cifar-10-batches-py")
    hf_format_dir = os.path.join(data_dir, "plain_text")

    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
    ])
    val_transform = T.ToTensor()

    if os.path.isdir(tv_format_dir):
        train_ds = tvdatasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=train_transform,
        )
        val_ds = tvdatasets.CIFAR10(
            root=data_dir, train=False, download=False, transform=val_transform,
        )
    elif os.path.isdir(hf_format_dir):
        train_ds = CIFAR10ParquetDataset(data_dir, split="train", transform=train_transform)
        val_ds   = CIFAR10ParquetDataset(data_dir, split="test",  transform=val_transform)
    else:
        raise FileNotFoundError(
            f"CIFAR-10 数据未找到：\n"
            f"  torchvision 格式路径不存在：{tv_format_dir}\n"
            f"  HuggingFace 格式路径不存在：{hf_format_dir}\n"
            f"请确认 --data-dir 参数，或在该目录下下载数据。"
        )

    return _make_loaders(train_ds, val_ds, batch_size, num_workers, distributed)


def get_div2k_loaders(
    data_dir: str,
    batch_size: int,
    crop_size: int = 256,
    num_workers: int = 4,
    distributed: bool = False,
    use_lmdb: bool = True,
    *,
    train_lmdb_path: str | None = None,
    val_lmdb_path: str | None = None,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    """DIV2K DataLoader（优先 LMDB，否则图像文件）。

    默认若存在以下 LMDB 则使用：
      <data_dir>/DIV2K_train_HR/train-{crop_size}.lmdb
      <data_dir>/DIV2K_valid_HR/valid-{crop_size}.lmdb

    若同时传入 ``train_lmdb_path`` 与 ``val_lmdb_path``，则改用这两条路径
    （用于训练集与验证集不在同一目录树下的情况，例如 DF2K + DIV2K valid）。
    可选 ``<path>.keys.cache`` 可加速 key 加载。
    """
    train_hr = os.path.join(data_dir, "DIV2K_train_HR")
    valid_hr = os.path.join(data_dir, "DIV2K_valid_HR")
    default_train_lmdb = os.path.join(train_hr, f"train-{crop_size}.lmdb")
    default_valid_lmdb = os.path.join(valid_hr, f"valid-{crop_size}.lmdb")
    default_train_cache = os.path.join(train_hr, f"train-{crop_size}.lmdb.keys.cache")
    default_valid_cache = os.path.join(valid_hr, f"valid-{crop_size}.lmdb.keys.cache")

    if train_lmdb_path is not None or val_lmdb_path is not None:
        if train_lmdb_path is None or val_lmdb_path is None:
            raise ValueError(
                "DIV2K：train_lmdb_path 与 val_lmdb_path 必须同时指定或同时省略。"
            )
        train_lmdb = train_lmdb_path
        valid_lmdb = val_lmdb_path
        train_cache = f"{train_lmdb}.keys.cache"
        valid_cache = f"{valid_lmdb}.keys.cache"
        explicit_lmdb = True
    else:
        train_lmdb = default_train_lmdb
        valid_lmdb = default_valid_lmdb
        train_cache = default_train_cache
        valid_cache = default_valid_cache
        explicit_lmdb = False

    lmdb_exists = (
        (os.path.isdir(train_lmdb) or os.path.isfile(train_lmdb))
        and (os.path.isdir(valid_lmdb) or os.path.isfile(valid_lmdb))
    )
    if explicit_lmdb and not lmdb_exists:
        raise FileNotFoundError(
            "DIV2K：显式指定的 LMDB 路径不完整或不存在：\n"
            f"  train: {train_lmdb!r} 存在={os.path.isdir(train_lmdb) or os.path.isfile(train_lmdb)}\n"
            f"  val:   {valid_lmdb!r} 存在={os.path.isdir(valid_lmdb) or os.path.isfile(valid_lmdb)}"
        )
    use_lmdb = use_lmdb and lmdb_exists

    train_cache_file = train_cache if os.path.isfile(train_cache) else None
    valid_cache_file = valid_cache if os.path.isfile(valid_cache) else None

    if use_lmdb:
        try:
            train_ds = DIV2KLMDBDataset(
                train_lmdb,
                crop_size=crop_size,
                split="train",
                keys_cache_path=train_cache_file,
            )
            val_ds = DIV2KLMDBDataset(
                valid_lmdb,
                crop_size=crop_size,
                split="valid",
                keys_cache_path=valid_cache_file,
            )
            # LMDB 样本数过少时回退到图像，避免 drop_last 导致 0 batch
            if len(train_ds) < batch_size:
                train_ds = DIV2KDataset(data_dir, crop_size=crop_size, split="train")
                val_ds = DIV2KDataset(data_dir, crop_size=crop_size, split="valid")
        except (ValueError, OSError, ImportError) as e:
            import warnings
            warnings.warn(f"LMDB 加载失败 ({e})，回退到图像格式", stacklevel=2)
            train_ds = DIV2KDataset(data_dir, crop_size=crop_size, split="train")
            val_ds = DIV2KDataset(data_dir, crop_size=crop_size, split="valid")
    else:
        train_ds = DIV2KDataset(data_dir, crop_size=crop_size, split="train")
        val_ds = DIV2KDataset(data_dir, crop_size=crop_size, split="valid")

    return _make_loaders(train_ds, val_ds, batch_size, num_workers, distributed)
