import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    from PIL import Image
except ImportError:
    Image = None


IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy", ".pt", ".pth"}


def _list_valid_files(folder: Path) -> Dict[str, Path]:
    files = {}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rel_key = p.relative_to(folder).with_suffix("").as_posix()
            files[rel_key] = p
    return files


def build_yeseg_samples(root: str, optical_dir="optical", sar_dir="sar", mask_dir="label"):
    root = Path(root)
    opt_files = _list_valid_files(root / optical_dir)
    sar_files = _list_valid_files(root / sar_dir)
    mask_files = _list_valid_files(root / mask_dir)

    def sort_key(name: str):
        return tuple((0, int(part)) if part.isdigit() else (1, part) for part in name.split("/"))

    names = sorted(set(opt_files) & set(sar_files) & set(mask_files), key=sort_key)
    return [(opt_files[n], sar_files[n], mask_files[n]) for n in names]


def build_pohang_samples(root: str, scene: str):
    scene_root = Path(root) / scene
    opt_files = _list_valid_files(scene_root / "GF2")
    sar_files = _list_valid_files(scene_root / "GF3")
    mask_files = _list_valid_files(scene_root / "label")

    names = sorted(set(opt_files) & set(sar_files) & set(mask_files))
    return [(opt_files[n], sar_files[n], mask_files[n]) for n in names]


def split_samples(samples, val_ratio=0.2, seed=42):
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_count = max(1, int(len(samples) * val_ratio)) if len(samples) > 1 else 0
    val_idx = set(indices[:val_count])
    train_samples = [samples[i] for i in range(len(samples)) if i not in val_idx]
    val_samples = [samples[i] for i in range(len(samples)) if i in val_idx]
    return train_samples, val_samples


def _normalize_split_key(raw: str) -> str:
    key = str(raw).strip().replace("\\", "/")
    if len(key) == 0:
        return ""
    key = Path(key).with_suffix("").as_posix()
    while key.startswith("./"):
        key = key[2:]
    if key.startswith("/"):
        key = key[1:]
    return key


def _read_split_keys(split_file: Path) -> List[str]:
    keys: List[str] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            key = _normalize_split_key(line)
            if len(key) > 0:
                keys.append(key)
    return keys


def split_samples_by_txt(
    samples: List[Tuple[Path, Path, Path]],
    root: str,
    optical_dir: str = "optical",
    train_split_file: str = "train.txt",
    val_split_file: str = "val.txt",
):
    root = Path(root)
    train_path = root / str(train_split_file)
    val_path = root / str(val_split_file)

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"split txt missing: train={train_path.exists()} ({train_path}), "
            f"val={val_path.exists()} ({val_path})"
        )

    opt_root = root / optical_dir

    sample_by_key: Dict[str, Tuple[Path, Path, Path]] = {}
    stem_to_keys: Dict[str, List[str]] = {}
    for s in samples:
        opt_path = s[0]
        try:
            key = opt_path.relative_to(opt_root).with_suffix("").as_posix()
        except Exception:
            key = opt_path.with_suffix("").name
        key = _normalize_split_key(key)
        sample_by_key[key] = s
        stem = Path(key).name
        if stem not in stem_to_keys:
            stem_to_keys[stem] = []
        stem_to_keys[stem].append(key)

    def resolve(split_name: str, keys: List[str]):
        resolved_samples: List[Tuple[Path, Path, Path]] = []
        resolved_keys: List[str] = []
        missing_keys: List[str] = []
        seen = set()

        for key in keys:
            if key in seen:
                continue
            item = sample_by_key.get(key, None)
            if item is None:
                stem = Path(key).name
                candidates = stem_to_keys.get(stem, [])
                if len(candidates) == 1:
                    key = candidates[0]
                    item = sample_by_key[key]
            if item is None:
                missing_keys.append(key)
                continue

            seen.add(key)
            resolved_keys.append(key)
            resolved_samples.append(item)

        if len(missing_keys) > 0:
            print(
                f"[Warn] {split_name} has {len(missing_keys)} unmatched keys, "
                f"example={missing_keys[:10]}"
            )

        return resolved_samples, resolved_keys

    train_keys = _read_split_keys(train_path)
    val_keys = _read_split_keys(val_path)

    train_samples, train_used_keys = resolve("train split", train_keys)
    val_samples, val_used_keys = resolve("val split", val_keys)

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError(
            f"split from txt produced empty set: train={len(train_samples)}, val={len(val_samples)}"
        )

    overlap = set(train_used_keys) & set(val_used_keys)
    if len(overlap) > 0:
        overlap = set(overlap)
        val_samples_clean = []
        for s, k in zip(val_samples, val_used_keys):
            if k not in overlap:
                val_samples_clean.append(s)
        if len(val_samples_clean) > 0:
            print(f"[Warn] split txt overlap found ({len(overlap)} keys), removed overlap from val.")
            val_samples = val_samples_clean
        else:
            print(f"[Warn] split txt overlap found ({len(overlap)} keys), keep original val for safety.")

    return train_samples, val_samples


def sanitize_mask_labels(mask: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    valid = (mask >= 0) & (mask < int(num_classes))
    if valid.all():
        return mask
    out = mask.clone()
    out[~valid] = int(ignore_index)
    return out


def remap_mask_labels(mask: torch.Tensor, class_id_map: Dict[int, int], ignore_index: int) -> torch.Tensor:
    if not class_id_map:
        return mask
    out = torch.full_like(mask, int(ignore_index))
    for old_cls, new_cls in class_id_map.items():
        out[mask == int(old_cls)] = int(new_cls)
    return out


class MultimodalSegDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, Path, Path]],
        transform=None,
        ignore_index: int = 255,
        normalize_opt: bool = True,
        normalize_sar: bool = True,
        normalize_stage: str = "pre",
        sar_log_transform: bool = False,
        num_classes: int = 8,
        sanitize_labels: bool = True,
        class_id_map: Dict[int, int] = None,
        include_ignore_as_class: bool = False,
        ignore_as_class_id: int = None,
    ):
        super().__init__()
        self.samples = samples
        self.transform = transform
        self.ignore_index = int(ignore_index)
        self.normalize_opt = normalize_opt
        self.normalize_sar = normalize_sar
        self.normalize_stage = str(normalize_stage).lower()
        if self.normalize_stage not in {"pre", "post"}:
            raise ValueError(f"normalize_stage must be 'pre' or 'post', got: {normalize_stage}")
        self.sar_log_transform = bool(sar_log_transform)
        self.num_classes = int(num_classes)
        self.sanitize_labels = bool(sanitize_labels)
        self.include_ignore_as_class = bool(include_ignore_as_class)
        self.ignore_as_class_id = None if ignore_as_class_id is None else int(ignore_as_class_id)
        if class_id_map is None:
            self.class_id_map = {}
        else:
            self.class_id_map = {int(k): int(v) for k, v in class_id_map.items()}

    def __len__(self):
        return len(self.samples)

    def _read_raster(self, path: Path) -> np.ndarray:
        ext = path.suffix.lower()

        if ext == ".npy":
            arr = np.load(path)
        elif ext in [".pt", ".pth"]:
            arr = torch.load(path)
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            else:
                arr = np.asarray(arr)
        elif ext in [".tif", ".tiff"]:
            if rasterio is None:
                raise ImportError("Please install rasterio: pip install rasterio")
            with rasterio.open(path) as src:
                arr = src.read()
        elif ext in [".png", ".jpg", ".jpeg"]:
            if Image is None:
                raise ImportError("Please install pillow: pip install pillow")
            arr = np.array(Image.open(path))
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        return arr

    def _to_chw_float(self, arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            if arr.shape[0] > 16 and arr.shape[-1] <= 16:
                arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported image ndim={arr.ndim}")

        arr = np.ascontiguousarray(arr)
        return torch.as_tensor(arr.copy(), dtype=torch.float32)

    def _to_hw_long(self, arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                raise ValueError("Mask should be single-channel")

        arr = np.ascontiguousarray(arr)
        return torch.as_tensor(arr.copy(), dtype=torch.long)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[0]
        x_flat = x.view(c, -1)
        mean = x_flat.mean(dim=1, keepdim=True)
        std = x_flat.std(dim=1, keepdim=True).clamp_min(1e-6)
        return ((x_flat - mean) / std).view_as(x)

    def _normalize_modalities(self, opt: torch.Tensor, sar: torch.Tensor):
        if self.normalize_opt:
            opt = self._normalize(opt)

        if self.normalize_sar:
            if self.sar_log_transform:
                sar = torch.log1p(torch.clamp_min(sar, 0.0))
            sar = self._normalize(sar)

        return opt, sar

    def remap_mask(self, mask: torch.Tensor) -> torch.Tensor:
        orig_ignore = (mask == int(self.ignore_index))
        if self.class_id_map:
            out = remap_mask_labels(mask, self.class_id_map, self.ignore_index)
        else:
            out = mask

        if self.include_ignore_as_class and (self.ignore_as_class_id is not None):
            if 0 <= int(self.ignore_as_class_id) < int(self.num_classes):
                out = out.clone()
                out[orig_ignore] = int(self.ignore_as_class_id)
        return out

    def __getitem__(self, idx: int):
        opt_path, sar_path, mask_path = self.samples[idx]
        opt = self._to_chw_float(self._read_raster(opt_path))
        sar = self._to_chw_float(self._read_raster(sar_path))
        mask = self._to_hw_long(self._read_raster(mask_path))
        mask = self.remap_mask(mask)

        if self.normalize_stage == "pre":
            opt, sar = self._normalize_modalities(opt, sar)
        if self.sanitize_labels:
            mask = sanitize_mask_labels(mask, self.num_classes, self.ignore_index)

        sample = {
            "opt": opt.contiguous(),
            "sar": sar.contiguous(),
            "mask": mask.contiguous(),
            "name": opt_path.stem,
        }
        if self.transform is not None:
            sample = self.transform(sample)

        if self.normalize_stage == "post":
            opt = sample["opt"]
            sar = sample["sar"]
            opt, sar = self._normalize_modalities(opt, sar)
            sample["opt"] = opt.contiguous()
            sample["sar"] = sar.contiguous()

        # Ensure resizable tensor storage for worker-side collation.
        sample["opt"] = sample["opt"].contiguous().clone()
        sample["sar"] = sample["sar"].contiguous().clone()
        sample["mask"] = sample["mask"].contiguous().clone().long()
        if self.sanitize_labels:
            sample["mask"] = sanitize_mask_labels(sample["mask"], self.num_classes, self.ignore_index)
        return sample


class RandomFlipRotate:
    def __init__(self, flip_prob=0.5, rotate_prob=0.5):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

    def __call__(self, sample):
        opt, sar, mask = sample["opt"], sample["sar"], sample["mask"]

        if torch.rand(1).item() < self.flip_prob:
            opt = torch.flip(opt, dims=[2])
            sar = torch.flip(sar, dims=[2])
            mask = torch.flip(mask, dims=[1])

        if torch.rand(1).item() < self.flip_prob:
            opt = torch.flip(opt, dims=[1])
            sar = torch.flip(sar, dims=[1])
            mask = torch.flip(mask, dims=[0])

        if torch.rand(1).item() < self.rotate_prob:
            k = torch.randint(0, 4, (1,)).item()
            opt = torch.rot90(opt, k, dims=[1, 2])
            sar = torch.rot90(sar, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[0, 1])

        sample["opt"] = opt.contiguous()
        sample["sar"] = sar.contiguous()
        sample["mask"] = mask.contiguous()
        return sample


class OpticalColorJitter:
    def __init__(
        self,
        prob=0.5,
        brightness=0.2,
        contrast=0.2,
        noise_std=0.01,
    ):
        self.prob = prob
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, sample):
        if torch.rand(1).item() > self.prob:
            return sample

        opt = sample["opt"]
        if opt.dim() != 3:
            return sample

        if self.brightness > 0:
            brightness_factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.brightness
            opt = opt * brightness_factor

        if self.contrast > 0:
            mean = opt.mean(dim=(1, 2), keepdim=True)
            contrast_factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.contrast
            opt = (opt - mean) * contrast_factor + mean

        if self.noise_std > 0:
            opt = opt + torch.randn_like(opt) * self.noise_std

        sample["opt"] = opt.contiguous()
        return sample


class SARNoiseAug:
    def __init__(
        self,
        prob=0.5,
        noise_std=0.03,
        speckle_std=0.05,
    ):
        self.prob = prob
        self.noise_std = noise_std
        self.speckle_std = speckle_std

    def __call__(self, sample):
        if torch.rand(1).item() > self.prob:
            return sample

        sar = sample["sar"]
        if sar.dim() != 3:
            return sample

        if self.noise_std > 0:
            sar = sar + torch.randn_like(sar) * self.noise_std

        if self.speckle_std > 0:
            speckle = 1.0 + torch.randn_like(sar) * self.speckle_std
            sar = sar * speckle

        sample["sar"] = sar.contiguous()
        return sample


class RandomCropPair:
    def __init__(self, crop_size=512, ignore_index=255):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_h, self.crop_w = crop_size
        self.ignore_index = ignore_index

    def _safe_pad_image(self, x, pad_w, pad_h):
        _, h, w = x.shape
        if pad_h == 0 and pad_w == 0:
            return x

        can_reflect = (h > 1 and w > 1 and pad_h < h and pad_w < w)
        if can_reflect:
            return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    def _safe_pad_mask(self, x, pad_w, pad_h):
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=self.ignore_index)

    def __call__(self, sample):
        opt, sar, mask = sample["opt"], sample["sar"], sample["mask"]
        _, h, w = opt.shape

        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            opt = self._safe_pad_image(opt, pad_w, pad_h)
            sar = self._safe_pad_image(sar, pad_w, pad_h)
            mask = self._safe_pad_mask(mask, pad_w, pad_h)
            _, h, w = opt.shape

        crop_h = min(self.crop_h, h)
        crop_w = min(self.crop_w, w)

        top = 0 if h == crop_h else torch.randint(0, h - crop_h + 1, (1,)).item()
        left = 0 if w == crop_w else torch.randint(0, w - crop_w + 1, (1,)).item()

        sample["opt"] = opt[:, top:top + crop_h, left:left + crop_w].contiguous()
        sample["sar"] = sar[:, top:top + crop_h, left:left + crop_w].contiguous()
        sample["mask"] = mask[top:top + crop_h, left:left + crop_w].contiguous()
        return sample


class ClassAwareRandomCrop:
    def __init__(self, crop_size=512, target_class=7, min_pixels=20, max_try=20, ignore_index=255):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_h, self.crop_w = crop_size
        self.target_class = target_class
        self.min_pixels = min_pixels
        self.max_try = max_try
        self.ignore_index = ignore_index

    def _safe_pad_image(self, x, pad_w, pad_h):
        _, h, w = x.shape
        if pad_h == 0 and pad_w == 0:
            return x

        can_reflect = (h > 1 and w > 1 and pad_h < h and pad_w < w)
        if can_reflect:
            return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    def _safe_pad_mask(self, x, pad_w, pad_h):
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=self.ignore_index)

    def __call__(self, sample):
        opt, sar, mask = sample["opt"], sample["sar"], sample["mask"]
        _, h, w = opt.shape

        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            opt = self._safe_pad_image(opt, pad_w, pad_h)
            sar = self._safe_pad_image(sar, pad_w, pad_h)
            mask = self._safe_pad_mask(mask, pad_w, pad_h)
            _, h, w = opt.shape

        crop_h = self.crop_h
        crop_w = self.crop_w
        best_top, best_left = 0, 0
        best_count = -1

        for _ in range(self.max_try):
            top = 0 if h == crop_h else torch.randint(0, h - crop_h + 1, (1,)).item()
            left = 0 if w == crop_w else torch.randint(0, w - crop_w + 1, (1,)).item()
            crop_mask = mask[top:top + crop_h, left:left + crop_w]
            count = int((crop_mask == self.target_class).sum().item())

            if count > best_count:
                best_count = count
                best_top, best_left = top, left

            if count >= self.min_pixels:
                break

        top, left = best_top, best_left
        sample["opt"] = opt[:, top:top + crop_h, left:left + crop_w].contiguous()
        sample["sar"] = sar[:, top:top + crop_h, left:left + crop_w].contiguous()
        sample["mask"] = mask[top:top + crop_h, left:left + crop_w].contiguous()
        return sample


class ProbabilisticClassAwareCrop:
    def __init__(
        self,
        crop_size=512,
        target_class=7,
        class_aware_prob=0.5,
        min_pixels=20,
        max_try=20,
        ignore_index=255,
    ):
        self.class_aware_prob = class_aware_prob
        self.random_crop = RandomCropPair(crop_size=crop_size, ignore_index=ignore_index)
        self.class_aware_crop = ClassAwareRandomCrop(
            crop_size=crop_size,
            target_class=target_class,
            min_pixels=min_pixels,
            max_try=max_try,
            ignore_index=ignore_index,
        )

    def __call__(self, sample):
        if torch.rand(1).item() < self.class_aware_prob:
            return self.class_aware_crop(sample)
        return self.random_crop(sample)


class TargetClassCopyPaste:
    def __init__(
        self,
        samples,
        target_class=7,
        prob=0.5,
        min_pixels=10,
        max_paste=1,
        patch_size=128,
        ignore_index=255,
        normalize_opt=True,
        normalize_sar=True,
        cache_size=0,
        num_classes=8,
        sanitize_labels=True,
        source_sampling_power=0.5,
        class_id_map: Dict[int, int] = None,
        include_ignore_as_class: bool = False,
        ignore_as_class_id: int = None,
    ):
        self.samples = samples
        self.target_class = target_class
        self.prob = prob
        self.min_pixels = min_pixels
        self.max_paste = max_paste
        self.patch_size = patch_size
        self.ignore_index = ignore_index
        self.normalize_opt = normalize_opt
        self.normalize_sar = normalize_sar
        self.cache_size = max(0, int(cache_size))
        self.num_classes = int(num_classes)
        self.sanitize_labels = bool(sanitize_labels)
        self.source_sampling_power = float(source_sampling_power)
        self.include_ignore_as_class = bool(include_ignore_as_class)
        self.ignore_as_class_id = None if ignore_as_class_id is None else int(ignore_as_class_id)
        if class_id_map is None:
            self.class_id_map = {}
        else:
            self.class_id_map = {int(k): int(v) for k, v in class_id_map.items()}
        self._cache = OrderedDict()

        self.rare_indices, self.rare_weights = self._build_rare_indices()
        print(
            f"[CopyPaste] target_class={target_class}, "
            f"rare_samples={len(self.rare_indices)}/{len(samples)}"
        )

    def _read_raster(self, path):
        ext = path.suffix.lower()

        if ext == ".npy":
            arr = np.load(path)
        elif ext in [".pt", ".pth"]:
            arr = torch.load(path)
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            else:
                arr = np.asarray(arr)
        elif ext in [".tif", ".tiff"]:
            if rasterio is None:
                raise ImportError("Please install rasterio: pip install rasterio")
            with rasterio.open(path) as src:
                arr = src.read()
        elif ext in [".png", ".jpg", ".jpeg"]:
            if Image is None:
                raise ImportError("Please install pillow: pip install pillow")
            arr = np.array(Image.open(path))
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return arr

    def _to_chw_float(self, arr):
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            if arr.shape[0] > 16 and arr.shape[-1] <= 16:
                arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported image ndim={arr.ndim}")

        arr = np.ascontiguousarray(arr)
        return torch.as_tensor(arr.copy(), dtype=torch.float32)

    def _to_hw_long(self, arr):
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                raise ValueError("Mask should be single-channel")

        arr = np.ascontiguousarray(arr)
        return torch.as_tensor(arr.copy(), dtype=torch.long)

    def _normalize(self, x):
        c = x.shape[0]
        x_flat = x.view(c, -1)
        mean = x_flat.mean(dim=1, keepdim=True)
        std = x_flat.std(dim=1, keepdim=True).clamp_min(1e-6)
        return ((x_flat - mean) / std).view_as(x)

    def _maybe_sanitize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        orig_ignore = (mask == int(self.ignore_index))
        if self.class_id_map:
            mask = remap_mask_labels(mask, self.class_id_map, self.ignore_index)
        if self.include_ignore_as_class and (self.ignore_as_class_id is not None):
            if 0 <= int(self.ignore_as_class_id) < int(self.num_classes):
                mask = mask.clone()
                mask[orig_ignore] = int(self.ignore_as_class_id)
        if not self.sanitize_labels:
            return mask
        return sanitize_mask_labels(mask, self.num_classes, self.ignore_index)

    def _build_rare_indices(self):
        rare_indices = []
        rare_weights = []

        for idx, (_, _, mask_path) in enumerate(self.samples):
            mask_arr = self._read_raster(mask_path)
            mask = self._to_hw_long(mask_arr)
            mask = self._maybe_sanitize_mask(mask)
            rare_pixels = int((mask == self.target_class).sum().item())

            if rare_pixels >= self.min_pixels:
                rare_indices.append(idx)
                rare_weights.append(max(float(rare_pixels), 1.0) ** self.source_sampling_power)

        return rare_indices, rare_weights

    def _evict_cache_if_needed(self):
        if self.cache_size <= 0:
            return
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _load_source_sample(self, idx):
        if idx in self._cache:
            opt, sar, mask = self._cache.pop(idx)
            self._cache[idx] = (opt, sar, mask)
            return opt.clone(), sar.clone(), mask.clone()

        opt_path, sar_path, mask_path = self.samples[idx]
        opt = self._to_chw_float(self._read_raster(opt_path))
        sar = self._to_chw_float(self._read_raster(sar_path))
        mask = self._to_hw_long(self._read_raster(mask_path))
        mask = self._maybe_sanitize_mask(mask)

        if self.normalize_opt:
            opt = self._normalize(opt)
        if self.normalize_sar:
            sar = self._normalize(sar)

        opt = opt.contiguous()
        sar = sar.contiguous()
        mask = mask.contiguous()

        if self.cache_size > 0:
            self._cache[idx] = (opt.clone(), sar.clone(), mask.clone())
            self._evict_cache_if_needed()

        return opt, sar, mask

    def _sample_source_patch(self, src_opt, src_sar, src_mask, max_retry=6):
        _, h, w = src_opt.shape
        ys, xs = torch.where(src_mask == self.target_class)
        if ys.numel() == 0:
            return None

        patch = min(self.patch_size, h, w)
        for _ in range(max_retry):
            pix_idx = torch.randint(0, ys.numel(), (1,)).item()
            cy = int(ys[pix_idx].item())
            cx = int(xs[pix_idx].item())

            y1 = max(min(cy - patch // 2, h - patch), 0)
            x1 = max(min(cx - patch // 2, w - patch), 0)
            y2 = y1 + patch
            x2 = x1 + patch

            crop_mask = src_mask[y1:y2, x1:x2]
            target_pixels = int((crop_mask == self.target_class).sum().item())
            if target_pixels < self.min_pixels:
                continue

            crop_opt = src_opt[:, y1:y2, x1:x2]
            crop_sar = src_sar[:, y1:y2, x1:x2]
            return crop_opt, crop_sar, crop_mask

        return None

    def _sample_source_index(self):
        if not self.rare_indices:
            return None
        if self.rare_weights and len(self.rare_weights) == len(self.rare_indices):
            return random.choices(self.rare_indices, weights=self.rare_weights, k=1)[0]
        return random.choice(self.rare_indices)

    def __call__(self, sample):
        if len(self.rare_indices) == 0:
            return sample
        if random.random() > self.prob:
            return sample

        opt, sar, mask = sample["opt"], sample["sar"], sample["mask"]
        _, h, w = opt.shape

        for _ in range(self.max_paste):
            src_idx = self._sample_source_index()
            if src_idx is None:
                break
            src_opt, src_sar, src_mask = self._load_source_sample(src_idx)

            patch_pack = self._sample_source_patch(src_opt, src_sar, src_mask)
            if patch_pack is None:
                continue

            crop_opt, crop_sar, crop_mask = patch_pack
            ph, pw = crop_mask.shape

            if ph > h or pw > w:
                continue

            top = 0 if h == ph else torch.randint(0, h - ph + 1, (1,)).item()
            left = 0 if w == pw else torch.randint(0, w - pw + 1, (1,)).item()

            paste_region = crop_mask == self.target_class
            if int(paste_region.sum().item()) < self.min_pixels:
                continue

            region_opt = opt[:, top:top + ph, left:left + pw]
            region_sar = sar[:, top:top + ph, left:left + pw]
            region_mask = mask[top:top + ph, left:left + pw]

            region_opt[:, paste_region] = crop_opt[:, paste_region]
            region_sar[:, paste_region] = crop_sar[:, paste_region]
            region_mask[paste_region] = self.target_class

            opt[:, top:top + ph, left:left + pw] = region_opt
            sar[:, top:top + ph, left:left + pw] = region_sar
            mask[top:top + ph, left:left + pw] = region_mask

        sample["opt"] = opt.contiguous()
        sample["sar"] = sar.contiguous()
        sample["mask"] = self._maybe_sanitize_mask(mask.contiguous())
        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
