import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import YESegConfig, PohangConfig
from dataset import (
    build_yeseg_samples,
    build_pohang_samples,
    split_samples,
    split_samples_by_txt,
    MultimodalSegDataset,
    Compose,
    RandomFlipRotate,
    RandomCropPair,
    ProbabilisticClassAwareCrop,
    TargetClassCopyPaste,
    OpticalColorJitter,
    SARNoiseAug,
    sanitize_mask_labels,
)
from models import GCFNet
from utils import (
    SegMetric,
    build_optimizer,
    build_criterion,
    build_ema,
    WarmupCosineScheduler,
)


def set_global_seed(seed: int, deterministic: bool = False):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train GCFNet on YESeg or Pohang")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.getenv("GCFNET_DATASET", "yeseg"),
        choices=["yeseg", "pohang"],
        help="Dataset profile to use.",
    )
    return parser.parse_args()


def build_cfg(dataset_name: str):
    name = str(dataset_name).strip().lower()
    if name == "pohang":
        return PohangConfig()
    return YESegConfig()


def build_tqdm_kwargs(cfg, leave, show):
    use_tqdm = bool(getattr(cfg, "use_tqdm", True))
    disable_non_tty = bool(getattr(cfg, "progress_disable_non_tty", True))
    disable = (not use_tqdm) or (disable_non_tty and (not sys.stdout.isatty())) or (not show)
    return {
        "leave": leave,
        "disable": disable,
        "ascii": bool(getattr(cfg, "progress_ascii", True)),
        "mininterval": float(getattr(cfg, "progress_mininterval", 0.5)),
        "dynamic_ncols": True,
        "ncols": int(getattr(cfg, "progress_ncols", 120)),
        "file": sys.stdout,
        "position": 0,
    }


def build_loader_kwargs(cfg, loader_workers, pin_memory):
    kwargs = {
        "num_workers": int(loader_workers),
        "pin_memory": bool(pin_memory),
    }
    if loader_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(cfg, "persistent_workers", True))
        kwargs["prefetch_factor"] = max(1, int(getattr(cfg, "prefetch_factor", 2)))
    return kwargs


def resolve_stage2_start_epoch(cfg):
    start_ratio = float(getattr(cfg, "stage2_start_ratio", 0.55))
    if start_ratio <= 1.0:
        return max(1, int(np.ceil(float(cfg.epochs) * start_ratio)))
    return max(1, int(start_ratio))


def set_backbone_trainable(model, trainable: bool):
    for name, param in model.named_parameters():
        if name.startswith(("opt_encoder.", "sar_encoder.")):
            param.requires_grad = bool(trainable)


def apply_stage2_overrides(cfg, train_transform_list):
    cfg.lovasz_weight = float(getattr(cfg, "stage2_lovasz_weight", cfg.lovasz_weight))
    cfg.rare_tversky_weight = float(getattr(cfg, "stage2_rare_tversky_weight", cfg.rare_tversky_weight))
    cfg.rare_margin_weight = float(getattr(cfg, "stage2_rare_margin_weight", cfg.rare_margin_weight))
    cfg.rare_confusion_weight = float(getattr(cfg, "stage2_rare_confusion_weight", cfg.rare_confusion_weight))
    cfg.deep_supervision_aux_weight = float(
        getattr(cfg, "stage2_deep_supervision_aux_weight", cfg.deep_supervision_aux_weight)
    )
    cfg.rare_confusion_negative_classes = tuple(
        getattr(cfg, "stage2_rare_confusion_negative_classes", cfg.rare_confusion_negative_classes)
    )

    stage2_class_aware_prob = float(
        getattr(cfg, "stage2_class_aware_crop_prob", getattr(cfg, "class_aware_crop_prob", 0.6))
    )
    stage2_copypaste_prob = float(
        getattr(cfg, "stage2_copypaste_prob", getattr(cfg, "copypaste_prob", 0.5))
    )
    stage2_copypaste_max_paste = int(
        getattr(cfg, "stage2_copypaste_max_paste", getattr(cfg, "copypaste_max_paste", 1))
    )

    for t in train_transform_list:
        if isinstance(t, ProbabilisticClassAwareCrop):
            t.class_aware_prob = stage2_class_aware_prob
        elif isinstance(t, TargetClassCopyPaste):
            t.prob = stage2_copypaste_prob
            t.max_paste = stage2_copypaste_max_paste


def _clone_state_dict(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _normalize_int_tuple(values):
    if values is None:
        return tuple()
    if not isinstance(values, (list, tuple, set)):
        return tuple()
    out = []
    for v in values:
        try:
            out.append(int(v))
        except Exception:
            continue
    return tuple(sorted(set(out)))


def _normalize_class_id_map(class_id_map):
    if not isinstance(class_id_map, dict):
        return {}
    out = {}
    for k, v in class_id_map.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def _cfg_class_setup_signature(cfg):
    return {
        "num_classes": int(getattr(cfg, "num_classes", -1)),
        "original_num_classes": int(getattr(cfg, "original_num_classes", getattr(cfg, "num_classes", -1))),
        "drop_classes": _normalize_int_tuple(getattr(cfg, "drop_classes", tuple())),
        "class_id_map": _normalize_class_id_map(getattr(cfg, "class_id_map", {})),
    }


def _extract_ckpt_class_setup_signature(ckpt):
    if not isinstance(ckpt, dict):
        return None
    meta = ckpt.get("meta", None)
    if not isinstance(meta, dict):
        return None

    if ("num_classes" not in meta) and ("class_id_map" not in meta) and ("drop_classes" not in meta):
        return None

    sig = {
        "num_classes": int(meta.get("num_classes", -1)),
        "original_num_classes": int(meta.get("original_num_classes", -1)),
        "drop_classes": _normalize_int_tuple(meta.get("drop_classes", tuple())),
        "class_id_map": _normalize_class_id_map(meta.get("class_id_map", {})),
    }
    return sig


def build_checkpoint_payload(model, ema, optimizer, epoch, best_miou, cfg):
    save_compact = bool(getattr(cfg, "save_compact_checkpoint", False))
    save_raw = bool(getattr(cfg, "save_model_raw_state", True)) and (not save_compact)
    save_ema_model = bool(getattr(cfg, "save_model_ema_state", True)) and (not save_compact)
    save_ema_shadow = bool(getattr(cfg, "save_ema_shadow_state", True)) and (not save_compact)
    save_optimizer = bool(getattr(cfg, "save_optimizer_state", True)) and (not save_compact)

    model_raw = None
    model_ema = None

    use_ema_for_save = bool(getattr(cfg, "save_use_ema_weights", True))
    if ema is not None and use_ema_for_save:
        ema.apply_shadow(model)
        model_for_eval = _clone_state_dict(model)
        if save_ema_model:
            model_ema = model_for_eval
        ema.restore(model)
        if save_raw:
            model_raw = _clone_state_dict(model)
    else:
        model_for_eval = _clone_state_dict(model)
        if save_raw:
            model_raw = model_for_eval

    payload = {
        "model": model_for_eval,
        "model_raw": model_raw,
        "model_ema": model_ema,
        "ema": ema.shadow if (ema is not None and save_ema_shadow) else None,
        "epoch": int(epoch),
        "best_miou": float(best_miou),
        "optimizer": optimizer.state_dict() if save_optimizer else None,
        "meta": {
            "dataset_name": str(getattr(cfg, "dataset_name", "")),
            **_cfg_class_setup_signature(cfg),
        },
    }
    return payload


def safe_torch_save(payload, path):
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    tmp_path = f"{path}.tmp"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


@torch.no_grad()
def sliding_window_inference(
    model,
    x_opt,
    x_sar,
    num_classes,
    window=512,
    stride=384,
):
    b, _, h, w = x_opt.shape
    device = x_opt.device

    output = torch.zeros((b, num_classes, h, w), device=device)
    count_map = torch.zeros((b, 1, h, w), device=device)

    if h <= window and w <= window:
        pred = model(x_opt, x_sar)
        if isinstance(pred, dict):
            pred = pred["main"]
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        return pred

    ys = list(range(0, max(h - window, 0) + 1, stride))
    xs = list(range(0, max(w - window, 0) + 1, stride))

    if len(ys) == 0 or ys[-1] != h - window:
        ys.append(max(h - window, 0))
    if len(xs) == 0 or xs[-1] != w - window:
        xs.append(max(w - window, 0))

    for y1 in ys:
        for x1 in xs:
            y2 = min(y1 + window, h)
            x2 = min(x1 + window, w)
            y1 = max(y2 - window, 0)
            x1 = max(x2 - window, 0)

            patch_opt = x_opt[:, :, y1:y2, x1:x2]
            patch_sar = x_sar[:, :, y1:y2, x1:x2]

            pred = model(patch_opt, patch_sar)
            if isinstance(pred, dict):
                pred = pred["main"]

            pred = F.interpolate(
                pred,
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            )

            output[:, :, y1:y2, x1:x2] += pred
            count_map[:, :, y1:y2, x1:x2] += 1.0

    return output / count_map.clamp_min(1.0)


@torch.no_grad()
def multi_scale_inference(
    model,
    x_opt,
    x_sar,
    num_classes,
    scales=(1.0,),
    window=512,
    stride=384,
    use_tta=False,
):
    _, _, h, w = x_opt.shape
    preds = []

    for s in scales:
        new_h = max(int(round(h * s)), 32)
        new_w = max(int(round(w * s)), 32)

        opt_scaled = F.interpolate(x_opt, size=(new_h, new_w), mode="bilinear", align_corners=False)
        sar_scaled = F.interpolate(x_sar, size=(new_h, new_w), mode="bilinear", align_corners=False)

        pred = sliding_window_inference(
            model,
            opt_scaled,
            sar_scaled,
            num_classes=num_classes,
            window=window,
            stride=stride,
        )
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        preds.append(pred)

        if use_tta:
            pred_h = sliding_window_inference(
                model,
                torch.flip(opt_scaled, dims=[3]),
                torch.flip(sar_scaled, dims=[3]),
                num_classes=num_classes,
                window=window,
                stride=stride,
            )
            pred_h = torch.flip(pred_h, dims=[3])
            pred_h = F.interpolate(pred_h, size=(h, w), mode="bilinear", align_corners=False)
            preds.append(pred_h)

            pred_v = sliding_window_inference(
                model,
                torch.flip(opt_scaled, dims=[2]),
                torch.flip(sar_scaled, dims=[2]),
                num_classes=num_classes,
                window=window,
                stride=stride,
            )
            pred_v = torch.flip(pred_v, dims=[2])
            pred_v = F.interpolate(pred_v, size=(h, w), mode="bilinear", align_corners=False)
            preds.append(pred_v)

    return torch.mean(torch.stack(preds, dim=0), dim=0)


@torch.no_grad()
def evaluate(model, loader, device, cfg, criterion, ema=None, epoch=None):
    model.eval()
    metric = SegMetric(
        cfg.num_classes,
        ignore_index=cfg.ignore_index,
        ignore_absent_classes=bool(getattr(cfg, "metric_ignore_absent_classes", False)),
    )
    total_loss = 0.0
    ema_applied = False
    print(f"[Eval] start validation: batches={len(loader)}")

    if ema is not None:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            ema.apply_shadow(model)
            ema_applied = True
        except torch.cuda.OutOfMemoryError:
            print("[Warn] OOM when applying EMA shadow for validation; fallback to current model weights.")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    progress = tqdm(
        loader,
        desc="Valid",
        **build_tqdm_kwargs(
            cfg=cfg,
            leave=False,
            show=bool(getattr(cfg, "show_val_progress", False)),
        ),
    )
    for batch in progress:
        x_opt = batch["opt"].to(device, non_blocking=True)
        x_sar = batch["sar"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        if cfg.use_sliding_window:
            eval_epoch = int(epoch or 0)
            tta_start_epoch = int(getattr(cfg, "eval_tta_start_epoch", 0))
            use_eval_tta = bool(getattr(cfg, "use_tta", False)) and eval_epoch >= tta_start_epoch
            eval_scales = tuple(getattr(cfg, "eval_ms_scales", getattr(cfg, "ms_scales", (1.0,))))
            pred = multi_scale_inference(
                model,
                x_opt,
                x_sar,
                num_classes=cfg.num_classes,
                scales=eval_scales,
                window=cfg.window_size,
                stride=cfg.window_stride,
                use_tta=use_eval_tta,
            )
            loss = criterion(pred, y)
            metric.update(pred, y)
        else:
            pred = model(x_opt, x_sar)
            loss = criterion(pred, y)
            metric.update(pred["main"] if isinstance(pred, dict) else pred, y)

        if torch.isfinite(loss):
            loss_item = float(loss.item())
        else:
            loss_item = 0.0
            print("[Warn] non-finite val loss encountered; set to 0 for averaging.")

        total_loss += loss_item
        progress.set_postfix(loss=f"{loss_item:.4f}")

    if ema_applied:
        ema.restore(model)

    scores = metric.get_scores()
    scores["val_loss"] = total_loss / max(len(loader), 1)
    return scores


@torch.no_grad()
def report_label_quality(train_set, num_classes, ignore_index, max_samples=0):
    invalid_pixels = 0
    total_pixels = 0
    invalid_hist = {}

    samples = train_set.samples if max_samples <= 0 else train_set.samples[:max_samples]
    for _, _, mask_path in samples:
        mask_arr = train_set._read_raster(mask_path)
        mask = train_set._to_hw_long(mask_arr)
        if hasattr(train_set, "remap_mask"):
            mask = train_set.remap_mask(mask)
        mask = mask.view(-1)
        total_pixels += int(mask.numel())
        bad = (mask < 0) | (mask >= num_classes)
        if not bad.any():
            continue

        invalid = mask[bad]
        invalid_pixels += int(invalid.numel())
        vals, cnts = torch.unique(invalid, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            invalid_hist[int(v)] = invalid_hist.get(int(v), 0) + int(c)

    invalid_ratio = invalid_pixels / max(total_pixels, 1)
    top_invalid = sorted(invalid_hist.items(), key=lambda x: -x[1])[:8]
    print(
        f"[Label quality] invalid_pixels={invalid_pixels}/{total_pixels} "
        f"({invalid_ratio:.6%}), top_invalid={top_invalid}"
    )


@torch.no_grad()
def compute_class_weights_from_samples(train_set, num_classes, ignore_index, device):
    hist = torch.zeros(num_classes, dtype=torch.float64)

    for _, _, mask_path in train_set.samples:
        mask_arr = train_set._read_raster(mask_path)
        mask = train_set._to_hw_long(mask_arr)
        if hasattr(train_set, "remap_mask"):
            mask = train_set.remap_mask(mask)
        mask = sanitize_mask_labels(mask, num_classes=num_classes, ignore_index=ignore_index)
        mask = mask.view(-1)
        mask = mask[(mask >= 0) & (mask < num_classes) & (mask != ignore_index)]
        if mask.numel() > 0:
            hist += torch.bincount(mask, minlength=num_classes).double()

    freq = hist / hist.sum().clamp_min(1.0)
    weights = 1.0 / torch.log(1.02 + freq)
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights.to(device=device, dtype=torch.float32)


def build_weighted_sampler(
    train_set,
    target_class=7,
    boost=6.0,
    min_pixels=20,
    ref_pixels=320,
    max_weight=6.0,
    target_ratio=0.25,
    target_ratio_mode="lower_bound",
    rare_weight_power=0.6,
):
    weights = []
    rare_pixels_list = []
    ref_pixels = max(int(ref_pixels), 1)
    max_weight = max(float(max_weight), 1e-3)
    boost = float(boost)
    min_pixels = max(int(min_pixels), 1)
    rare_weight_power = max(float(rare_weight_power), 1e-6)
    target_ratio = float(target_ratio)
    target_ratio_mode = str(target_ratio_mode).lower()

    print(f"Building WeightedRandomSampler for class {target_class}...")
    for _, _, mask_path in train_set.samples:
        mask_arr = train_set._read_raster(mask_path)
        mask = train_set._to_hw_long(mask_arr)
        if hasattr(train_set, "remap_mask"):
            mask = train_set.remap_mask(mask)
        mask = sanitize_mask_labels(mask, train_set.num_classes, train_set.ignore_index)
        rare_pixels = int((mask == target_class).sum().item())
        rare_pixels_list.append(rare_pixels)

    rare_flags = [rp >= min_pixels for rp in rare_pixels_list]
    n_rare = int(sum(rare_flags))
    n_common = int(len(rare_flags) - n_rare)

    base_rare_weight = boost
    if 0.0 < target_ratio < 1.0 and n_rare > 0 and n_common > 0:
        # Solve expected rare ratio under weighted sampling:
        # ratio = (n_rare*w) / (n_rare*w + n_common*1)
        required_weight = (target_ratio * n_common) / max((1.0 - target_ratio) * n_rare, 1e-8)
        if target_ratio_mode == "match_target":
            base_rare_weight = required_weight
        elif target_ratio_mode == "upper_bound":
            base_rare_weight = min(base_rare_weight, required_weight)
        else:
            # Legacy behavior: enforce a minimum boost.
            base_rare_weight = max(base_rare_weight, required_weight)

    base_rare_weight = float(min(max(base_rare_weight, 1e-3), max_weight))

    for rare_pixels, is_rare in zip(rare_pixels_list, rare_flags):
        if not is_rare:
            sample_w = 1.0
        else:
            ratio = min(float(rare_pixels) / float(ref_pixels), 1.0)
            ratio = ratio ** rare_weight_power
            sample_w = 1.0 + (base_rare_weight - 1.0) * ratio
            sample_w = min(sample_w, max_weight)
        weights.append(sample_w)

    weights = torch.DoubleTensor(weights)
    rare_weight_sum = float(weights[torch.tensor(rare_flags, dtype=torch.bool)].sum().item()) if n_rare > 0 else 0.0
    expected_rare_ratio = rare_weight_sum / max(float(weights.sum().item()), 1e-8)

    print(f"Rare-class samples: {n_rare}/{len(weights)}")
    print(f"Sampler weights: min={weights.min().item()}, max={weights.max().item()}")
    print(
        f"Sampler target_ratio={target_ratio:.3f}, "
        f"mode={target_ratio_mode}, "
        f"expected_rare_ratio={expected_rare_ratio:.3f}, "
        f"base_rare_weight={base_rare_weight:.3f}"
    )

    if n_rare == 0:
        print(
            f"[Warn] target_class={target_class} has zero eligible samples in training set; "
            "disable WeightedRandomSampler for this run."
        )
        return None

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def load_checkpoint_with_report(
    model,
    ckpt_path,
    device,
    strict=False,
    prefer_ema=False,
    expected_cfg=None,
    enforce_class_setup_match=False,
    allow_resume_without_meta=False,
):
    ckpt = torch.load(ckpt_path, map_location=device)
    source = "raw_checkpoint"
    state_dict = ckpt
    if isinstance(ckpt, dict):
        if prefer_ema and ("model_ema" in ckpt) and (ckpt["model_ema"] is not None):
            state_dict = ckpt["model_ema"]
            source = "model_ema"
        elif prefer_ema and ("ema" in ckpt) and isinstance(ckpt["ema"], dict) and len(ckpt["ema"]) > 0:
            state_dict = ckpt["ema"]
            source = "ema_shadow"
        elif "model" in ckpt:
            state_dict = ckpt["model"]
            source = "model"
        elif "model_raw" in ckpt:
            state_dict = ckpt["model_raw"]
            source = "model_raw"

    if enforce_class_setup_match and (expected_cfg is not None):
        expected_sig = _cfg_class_setup_signature(expected_cfg)
        ckpt_sig = _extract_ckpt_class_setup_signature(ckpt)
        class_drop_enabled = bool(getattr(expected_cfg, "enable_class_drop", False))

        if ckpt_sig is None:
            if class_drop_enabled and (not bool(allow_resume_without_meta)):
                raise RuntimeError(
                    "Checkpoint missing class-setup metadata, which is unsafe for class-drop resume. "
                    "Please train from scratch once under current class mapping, or set "
                    "allow_resume_without_class_meta=True to override."
                )
        else:
            mismatch_reasons = []
            if ckpt_sig["num_classes"] != expected_sig["num_classes"]:
                mismatch_reasons.append(
                    f"num_classes ckpt={ckpt_sig['num_classes']} cfg={expected_sig['num_classes']}"
                )
            if ckpt_sig["drop_classes"] != expected_sig["drop_classes"]:
                mismatch_reasons.append(
                    f"drop_classes ckpt={ckpt_sig['drop_classes']} cfg={expected_sig['drop_classes']}"
                )
            if ckpt_sig["class_id_map"] != expected_sig["class_id_map"]:
                mismatch_reasons.append(
                    f"class_id_map ckpt={ckpt_sig['class_id_map']} cfg={expected_sig['class_id_map']}"
                )

            if mismatch_reasons:
                reason_text = "; ".join(mismatch_reasons)
                raise RuntimeError(
                    "Checkpoint class setup mismatch, resume aborted: " + reason_text
                )

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    print("\n[Checkpoint check]")
    print(f"Missing keys   : {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    print(f"Loaded weights : {source}")
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError("Checkpoint does not fully match current model.")

    print(f"Loaded checkpoint: {ckpt_path}")
    return ckpt


def main():
    args = parse_args()
    cfg = build_cfg(args.dataset)
    print(f"[Config] dataset profile: {args.dataset}")
    set_global_seed(
        seed=getattr(cfg, "seed", 42),
        deterministic=bool(getattr(cfg, "deterministic", False)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_windows = os.name == "nt"
    loader_workers = 0 if is_windows else cfg.num_workers
    pin_memory = device.type == "cuda"

    normalize_opt = bool(getattr(cfg, "normalize_opt", True))
    normalize_sar = bool(getattr(cfg, "normalize_sar", True))
    train_normalize_stage = str(getattr(cfg, "train_normalize_stage", "pre")).lower()
    val_normalize_stage = str(getattr(cfg, "val_normalize_stage", "pre")).lower()
    train_sar_log_transform = bool(getattr(cfg, "train_sar_log_transform", False))
    val_sar_log_transform = bool(getattr(cfg, "val_sar_log_transform", train_sar_log_transform))
    print(
        f"Normalize opt/sar=({normalize_opt}/{normalize_sar}) | "
        f"train_stage={train_normalize_stage}, val_stage={val_normalize_stage}, "
        f"train_sar_log={train_sar_log_transform}"
    )
    print(
        f"Class setup: original={getattr(cfg, 'original_num_classes', cfg.num_classes)} "
        f"-> train={cfg.num_classes}, drop={tuple(getattr(cfg, 'drop_classes', tuple()))}, "
        f"map={getattr(cfg, 'class_id_map', {})}"
    )
    print(
        "Module switches: "
        f"OPT={bool(getattr(cfg, 'use_opt_modality', True))}, "
        f"SAR={bool(getattr(cfg, 'use_sar_modality', True))}, "
        f"GMF={bool(getattr(cfg, 'use_gmf_module', True))}, "
        f"GAM={bool(getattr(cfg, 'use_gam_module', True))}, "
        f"GLVRM={bool(getattr(cfg, 'use_glvrm', True))}, "
        f"ASPP={bool(getattr(cfg, 'use_aspp_context', True))}, "
        f"DeepSupervision={bool(getattr(cfg, 'deep_supervision', True))}"
    )

    print(f"Dataset root: {cfg.root}")
    print(f"Device: {device}")

    if str(getattr(cfg, "dataset_name", "")).lower() == "pohang":
        if bool(getattr(cfg, "use_official_split_txt", True)):
            scene_list = tuple(getattr(cfg, "pohang_split_scenes", tuple()))
            if len(scene_list) == 0:
                scene_list = (
                    str(getattr(cfg, "train_scene", "")).strip(),
                    str(getattr(cfg, "val_scene", "")).strip(),
                )
                scene_list = tuple([s for s in scene_list if len(s) > 0])
            if len(scene_list) == 0:
                raise RuntimeError(
                    "Pohang txt split requires non-empty `pohang_split_scenes` or train/val scenes."
                )

            all_samples = []
            for scene in scene_list:
                all_samples.extend(build_pohang_samples(cfg.root, scene))
            if len(all_samples) == 0:
                raise RuntimeError(
                    f"No Pohang samples found under root={cfg.root} for scenes={scene_list}."
                )

            train_split_file = getattr(cfg, "train_split_file", "train_pohang.txt")
            val_split_file = getattr(cfg, "val_split_file", "val_pohang.txt")
            train_samples, val_samples = split_samples_by_txt(
                samples=all_samples,
                root=cfg.root,
                optical_dir=".",
                train_split_file=train_split_file,
                val_split_file=val_split_file,
            )
            print(
                f"[Split] Pohang txt split: scenes={scene_list}, "
                f"train={train_split_file}, val={val_split_file}"
            )
        else:
            scene_list = tuple(getattr(cfg, "pohang_split_scenes", tuple()))
            if len(scene_list) == 0:
                scene_list = (
                    str(getattr(cfg, "train_scene", "")).strip(),
                    str(getattr(cfg, "val_scene", "")).strip(),
                )
                scene_list = tuple([s for s in scene_list if len(s) > 0])
            if len(scene_list) == 0:
                raise RuntimeError("Pohang random split requires non-empty `pohang_split_scenes` or train/val scenes.")

            all_samples = []
            for scene in scene_list:
                all_samples.extend(build_pohang_samples(cfg.root, scene))
            if len(all_samples) == 0:
                raise RuntimeError(
                    f"No Pohang samples found under root={cfg.root} for scenes={scene_list}."
                )
            train_samples, val_samples = split_samples(
                all_samples,
                val_ratio=getattr(cfg, "val_ratio", 0.2),
                seed=getattr(cfg, "seed", 42),
            )
            print(
                f"[Split] Pohang random split: scenes={scene_list}, "
                f"val_ratio={getattr(cfg, 'val_ratio', 0.2)}, seed={getattr(cfg, 'seed', 42)}"
            )
    else:
        all_samples = build_yeseg_samples(cfg.root, cfg.optical_dir, cfg.sar_dir, cfg.mask_dir)
        train_split_file = getattr(cfg, "train_split_file", "train.txt")
        val_split_file = getattr(cfg, "val_split_file", "val.txt")
        if not bool(getattr(cfg, "use_official_split_txt", True)):
            print("[Split] force txt split is enabled in train.py, ignore cfg.use_official_split_txt=False.")
        train_samples, val_samples = split_samples_by_txt(
            samples=all_samples,
            root=cfg.root,
            optical_dir=cfg.optical_dir,
            train_split_file=train_split_file,
            val_split_file=val_split_file,
        )
        print(f"[Split] using txt split: train={train_split_file}, val={val_split_file}")
    print(f"Total: {len(all_samples)} | Train: {len(train_samples)} | Val: {len(val_samples)}")

    if len(train_samples) == 0:
        raise RuntimeError(f"No training samples were found under '{cfg.root}'.")

    rare_class_id = int(getattr(cfg, "rare_class_id", -1))
    rare_class_valid = 0 <= rare_class_id < int(getattr(cfg, "num_classes", 0))
    if (rare_class_id >= 0) and (not rare_class_valid):
        print(
            f"[Warn] rare_class_id={rare_class_id} is out of range for num_classes={cfg.num_classes}; "
            "disable rare-class pipeline."
        )
    class_id_map = dict(getattr(cfg, "class_id_map", {}))
    include_ignore_as_class = bool(getattr(cfg, "include_ignore_label_as_class", False))
    ignore_as_class_id = getattr(cfg, "ignore_label_class_id", None)
    copypaste_target_class = int(getattr(cfg, "copypaste_target_class", -1))
    copypaste_target_valid = 0 <= copypaste_target_class < int(getattr(cfg, "num_classes", 0))
    use_class_aware_crop = bool(getattr(cfg, "use_class_aware_crop", False)) and rare_class_valid
    use_copypaste = bool(getattr(cfg, "use_copypaste", False)) and copypaste_target_valid
    use_weighted_sampler = bool(getattr(cfg, "use_weighted_sampler", False)) and rare_class_valid

    train_transform_list = []
    if cfg.enable_crop:
        if use_class_aware_crop:
            train_transform_list.append(
                ProbabilisticClassAwareCrop(
                    crop_size=cfg.crop_size,
                    target_class=rare_class_id,
                    class_aware_prob=getattr(cfg, "class_aware_crop_prob", 0.5),
                    min_pixels=getattr(cfg, "class_aware_min_pixels", 20),
                    max_try=getattr(cfg, "class_aware_max_try", 20),
                    ignore_index=cfg.ignore_index,
                )
            )
        else:
            train_transform_list.append(RandomCropPair(cfg.crop_size, ignore_index=cfg.ignore_index))

    if use_copypaste:
        copypaste_norm = (train_normalize_stage == "pre")
        train_transform_list.append(
            TargetClassCopyPaste(
                samples=train_samples,
                target_class=int(getattr(cfg, "copypaste_target_class", -1)),
                prob=getattr(cfg, "copypaste_prob", 0.5),
                min_pixels=getattr(cfg, "copypaste_min_pixels", 10),
                max_paste=getattr(cfg, "copypaste_max_paste", 1),
                patch_size=getattr(cfg, "copypaste_patch_size", 128),
                ignore_index=cfg.ignore_index,
                normalize_opt=normalize_opt and copypaste_norm,
                normalize_sar=normalize_sar and copypaste_norm,
                cache_size=getattr(cfg, "copypaste_cache_size", 0),
                num_classes=cfg.num_classes,
                sanitize_labels=getattr(cfg, "sanitize_labels", True),
                source_sampling_power=getattr(cfg, "copypaste_source_sampling_power", 0.5),
                class_id_map=class_id_map,
                include_ignore_as_class=include_ignore_as_class,
                ignore_as_class_id=ignore_as_class_id,
            )
        )

    train_transform_list.append(RandomFlipRotate(cfg.flip_prob, cfg.rotate_prob))
    train_transform_list.append(
        OpticalColorJitter(
            prob=getattr(cfg, "optical_jitter_prob", 0.5),
            brightness=getattr(cfg, "optical_brightness", 0.2),
            contrast=getattr(cfg, "optical_contrast", 0.2),
            noise_std=getattr(cfg, "optical_noise_std", 0.01),
        )
    )
    train_transform_list.append(
        SARNoiseAug(
            prob=getattr(cfg, "sar_noise_prob", 0.5),
            noise_std=getattr(cfg, "sar_noise_std", 0.03),
            speckle_std=getattr(cfg, "sar_speckle_std", 0.05),
        )
    )

    print("Train transforms:")
    for t in train_transform_list:
        print(" -", t.__class__.__name__)

    train_set = MultimodalSegDataset(
        samples=train_samples,
        transform=Compose(train_transform_list),
        ignore_index=cfg.ignore_index,
        normalize_opt=normalize_opt,
        normalize_sar=normalize_sar,
        normalize_stage=train_normalize_stage,
        sar_log_transform=train_sar_log_transform,
        num_classes=cfg.num_classes,
        sanitize_labels=getattr(cfg, "sanitize_labels", True),
        class_id_map=class_id_map,
        include_ignore_as_class=include_ignore_as_class,
        ignore_as_class_id=ignore_as_class_id,
    )
    val_set = MultimodalSegDataset(
        samples=val_samples,
        transform=None,
        ignore_index=cfg.ignore_index,
        normalize_opt=normalize_opt,
        normalize_sar=normalize_sar,
        normalize_stage=val_normalize_stage,
        sar_log_transform=val_sar_log_transform,
        num_classes=cfg.num_classes,
        sanitize_labels=getattr(cfg, "sanitize_labels", True),
        class_id_map=class_id_map,
        include_ignore_as_class=include_ignore_as_class,
        ignore_as_class_id=ignore_as_class_id,
    )

    if getattr(cfg, "report_label_quality", False):
        report_label_quality(
            train_set=train_set,
            num_classes=cfg.num_classes,
            ignore_index=cfg.ignore_index,
            max_samples=getattr(cfg, "label_quality_max_samples", 0),
        )

    if use_weighted_sampler:
        train_sampler = build_weighted_sampler(
            train_set,
            target_class=rare_class_id,
            boost=getattr(cfg, "rare_class_boost", 6.0),
            min_pixels=getattr(cfg, "min_rare_pixels", 20),
            ref_pixels=getattr(cfg, "rare_sample_ref_pixels", 320),
            max_weight=getattr(cfg, "rare_sample_max_weight", 6.0),
            target_ratio=getattr(cfg, "rare_target_ratio", 0.25),
            target_ratio_mode=getattr(cfg, "rare_target_ratio_mode", "lower_bound"),
            rare_weight_power=getattr(cfg, "rare_weight_power", 0.6),
        )
        train_shuffle = train_sampler is None
    else:
        train_sampler = None
        train_shuffle = True

    loader_kwargs = build_loader_kwargs(cfg, loader_workers=loader_workers, pin_memory=pin_memory)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        **loader_kwargs,
    )

    sample = train_set[0]
    print("opt shape:", sample["opt"].shape)
    print("sar shape:", sample["sar"].shape)
    print("mask shape:", sample["mask"].shape)

    model = GCFNet(
        num_classes=cfg.num_classes,
        opt_in_channels=sample["opt"].shape[0],
        sar_in_channels=sample["sar"].shape[0],
        decoder_dim=getattr(cfg, "decoder_dim", 192),
        deep_supervision=cfg.deep_supervision,
        eca_kernel_size=cfg.eca_kernel_size,
        use_opt_modality=getattr(cfg, "use_opt_modality", True),
        use_sar_modality=getattr(cfg, "use_sar_modality", True),
        use_gmf_module=getattr(cfg, "use_gmf_module", True),
        gam_reduction=cfg.gam_reduction,
        use_gam_module=getattr(cfg, "use_gam_module", True),
        gam_use_residual=cfg.gam_use_residual,
        gmf_use_residual_sum=cfg.gmf_use_residual_sum,
        pretrained_backbone=getattr(cfg, "pretrained_backbone", True),
        opt_pretrained_path=getattr(cfg, "opt_pretrained_path", ""),
        sar_pretrained_path=getattr(cfg, "sar_pretrained_path", ""),
        use_glvrm=getattr(cfg, "use_glvrm", True),
        glvrm_decoder_stages=getattr(cfg, "glvrm_decoder_stages", 1),
        use_aspp_context=getattr(cfg, "use_aspp_context", True),
        aspp_dilations=getattr(cfg, "aspp_dilations", (1, 3, 6)),
        aspp_dropout=getattr(cfg, "aspp_dropout", 0.05),
    ).to(device)

    resume_ckpt = None
    resume_enabled = bool(getattr(cfg, "resume", False))
    resume_path = str(getattr(cfg, "resume_path", ""))
    if resume_enabled and (not os.path.exists(resume_path)):
        print(f"[Warn] resume enabled but checkpoint not found: {resume_path}. Fallback to scratch.")
        resume_enabled = False

    if resume_enabled:
        try:
            resume_ckpt = load_checkpoint_with_report(
                model=model,
                ckpt_path=resume_path,
                device=device,
                strict=False,
                prefer_ema=bool(getattr(cfg, "resume_prefer_ema", True)),
                expected_cfg=cfg,
                enforce_class_setup_match=bool(getattr(cfg, "strict_resume_class_setup", True)),
                allow_resume_without_meta=bool(getattr(cfg, "allow_resume_without_class_meta", False)),
            )
            cfg.lr = float(getattr(cfg, "finetune_lr", cfg.lr))
            print(f"Train mode: finetune | lr={cfg.lr:.2e}")
        except RuntimeError as err:
            print(f"[Warn] {err}")
            print("[Warn] Resume disabled due to class-setup safety check. Fallback to scratch.")
            resume_enabled = False
            resume_ckpt = None
            print(f"Train mode: scratch | lr={cfg.lr:.2e}")
    else:
        print(f"Train mode: scratch | lr={cfg.lr:.2e}")

    freeze_backbone_epochs = int(getattr(cfg, "freeze_backbone_epochs", 0))
    backbone_frozen = (freeze_backbone_epochs > 0) and (not resume_enabled)
    if backbone_frozen:
        set_backbone_trainable(model, False)
        print(f"[Train] freeze optical/SAR encoders for first {freeze_backbone_epochs} epoch(s).")

    class_weights = None
    if getattr(cfg, "compute_class_weights", False):
        class_weights = compute_class_weights_from_samples(
            train_set=train_set,
            num_classes=cfg.num_classes,
            ignore_index=cfg.ignore_index,
            device=device,
        )
        class_weights = torch.clamp(
            class_weights,
            min=float(getattr(cfg, "class_weight_min", 0.5)),
            max=float(getattr(cfg, "class_weight_max", 5.0)),
        )
        boost_map = getattr(cfg, "class_weight_boost_map", {})
        if isinstance(boost_map, dict) and len(boost_map) > 0:
            for k, v in boost_map.items():
                cls_id = int(k)
                if 0 <= cls_id < class_weights.numel():
                    class_weights[cls_id] = class_weights[cls_id] * float(v)
            class_weights = torch.clamp(
                class_weights,
                min=float(getattr(cfg, "class_weight_min", 0.5)),
                max=float(getattr(cfg, "class_weight_max", 5.0)),
            )
            print(f"Class weight boost map applied: {boost_map}")
        print("Class weights:", class_weights)

    criterion = build_criterion(cfg, class_weights=class_weights)
    optimizer = build_optimizer(cfg, model)

    group_lrs = [pg["lr"] for pg in optimizer.param_groups]
    print(
        f"Optimizer param groups: {len(optimizer.param_groups)} | "
        f"lrs={[f'{lr:.2e}' for lr in group_lrs]}"
    )

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = cfg.epochs * steps_per_epoch
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=getattr(cfg, "warmup_epochs", 3),
        max_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        min_lr=getattr(cfg, "min_lr", 1e-6),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    ema = build_ema(cfg, model)

    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, cfg.save_name)
    last_save_name = str(getattr(cfg, "last_save_name", "last_gcfnet_final.pth"))
    last_save_path = os.path.join(cfg.save_dir, last_save_name)

    best_miou = -1.0
    if isinstance(resume_ckpt, dict) and ("best_miou" in resume_ckpt):
        try:
            best_miou = float(resume_ckpt["best_miou"])
            print(f"Resume best mIoU from checkpoint: {best_miou:.4f}")
        except Exception:
            best_miou = -1.0
    no_improve_epochs = 0
    early_stop_patience = int(getattr(cfg, "early_stop_patience", 0))
    early_stop_min_delta = float(getattr(cfg, "early_stop_min_delta", 0.0))
    skip_loss_if_gt = float(getattr(cfg, "skip_loss_if_gt", 0.0))
    global_step = 0
    stage2_enabled = bool(getattr(cfg, "enable_stage2_curriculum", False))
    stage2_applied = False
    stage2_start_epoch = resolve_stage2_start_epoch(cfg) if stage2_enabled else None
    early_stop_start_epoch = int(getattr(cfg, "early_stop_start_epoch", 0))
    if early_stop_start_epoch <= 0:
        if stage2_enabled:
            grace = int(getattr(cfg, "stage2_early_stop_grace_epochs", 2))
            early_stop_start_epoch = min(cfg.epochs, stage2_start_epoch + max(0, grace))
        else:
            early_stop_start_epoch = 1
    if stage2_enabled:
        print(f"Stage2 curriculum is enabled. Stage2 start epoch: {stage2_start_epoch}/{cfg.epochs}")
    if bool(getattr(cfg, "use_sliding_window", False)):
        train_eval_scales = tuple(getattr(cfg, "eval_ms_scales", getattr(cfg, "ms_scales", (1.0,))))
        print(
            "Eval inference: "
            f"sliding_window=True, window={getattr(cfg, 'window_size', '-')}, "
            f"stride={getattr(cfg, 'window_stride', '-')}, scales={train_eval_scales}, "
            f"tta={bool(getattr(cfg, 'use_tta', False))} from epoch {int(getattr(cfg, 'eval_tta_start_epoch', 0))}"
        )
    print(f"Early-stop starts from epoch: {early_stop_start_epoch}/{cfg.epochs}")

    for epoch in range(1, cfg.epochs + 1):
        if backbone_frozen and epoch > freeze_backbone_epochs:
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(cfg, model)
            scheduler = WarmupCosineScheduler(
                optimizer=optimizer,
                warmup_epochs=1,
                max_steps=max((cfg.epochs - epoch + 1) * steps_per_epoch, 1),
                steps_per_epoch=steps_per_epoch,
                min_lr=getattr(cfg, "min_lr", 1e-6),
            )
            global_step = 0
            backbone_frozen = False
            group_lrs = [pg["lr"] for pg in optimizer.param_groups]
            print(
                f"[Train] unfroze encoders at epoch {epoch}; "
                f"optimizer rebuilt with lrs={[f'{lr:.2e}' for lr in group_lrs]}"
            )

        if stage2_enabled and (not stage2_applied) and (epoch >= stage2_start_epoch):
            stage2_applied = True
            apply_stage2_overrides(cfg, train_transform_list)
            criterion = build_criterion(cfg, class_weights=class_weights)
            print(
                f"[Stage2] epoch={epoch} | "
                f"lovasz={cfg.lovasz_weight:.3f}, "
                f"tversky={cfg.rare_tversky_weight:.3f}, "
                f"margin={cfg.rare_margin_weight:.3f}, "
                f"confusion={cfg.rare_confusion_weight:.3f}"
            )

            if use_weighted_sampler:
                train_sampler = build_weighted_sampler(
                    train_set,
                    target_class=rare_class_id,
                    boost=getattr(cfg, "stage2_rare_class_boost", getattr(cfg, "rare_class_boost", 4.0)),
                    min_pixels=getattr(cfg, "min_rare_pixels", 20),
                    ref_pixels=getattr(cfg, "rare_sample_ref_pixels", 320),
                    max_weight=getattr(
                        cfg,
                        "stage2_rare_sample_max_weight",
                        getattr(cfg, "rare_sample_max_weight", 4.0),
                    ),
                    target_ratio=getattr(
                        cfg,
                        "stage2_rare_target_ratio",
                        getattr(cfg, "rare_target_ratio", 0.20),
                    ),
                    target_ratio_mode=getattr(
                        cfg,
                        "stage2_rare_target_ratio_mode",
                        getattr(cfg, "rare_target_ratio_mode", "lower_bound"),
                    ),
                    rare_weight_power=getattr(
                        cfg,
                        "stage2_rare_weight_power",
                        getattr(cfg, "rare_weight_power", 1.0),
                    ),
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=cfg.batch_size,
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                    **loader_kwargs,
                )

        model.train()
        total_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"Train {epoch:03d}",
            **build_tqdm_kwargs(cfg=cfg, leave=False, show=True),
        )

        for step, batch in enumerate(progress, start=1):
            global_step += 1

            x_opt = batch["opt"].to(device, non_blocking=True)
            x_sar = batch["sar"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            use_amp = bool(cfg.amp and device.type == "cuda")
            loss_value = 0.0
            pred = None
            loss = None

            try:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(x_opt, x_sar)
                    loss = criterion(pred, y)

                loss_value = float(loss.detach().item())
                if not torch.isfinite(loss):
                    print(f"[Warn] non-finite loss at epoch={epoch} step={step}, skip this batch.")
                    optimizer.zero_grad(set_to_none=True)
                    pred = None
                    loss = None
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                if skip_loss_if_gt > 0 and loss_value > skip_loss_if_gt:
                    print(
                        f"[Warn] abnormal loss={loss_value:.4f} > {skip_loss_if_gt:.4f} "
                        f"at epoch={epoch} step={step}, skip this batch."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    pred = None
                    loss = None
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

                scaler.scale(loss).backward()
                pred = None
                loss = None

            except torch.cuda.OutOfMemoryError:
                can_recover = bool(getattr(cfg, "auto_oom_recovery", True)) and device.type == "cuda"
                if (not can_recover):
                    raise

                micro_bs = max(1, int(getattr(cfg, "oom_micro_batch_size", 1)))
                if x_opt.shape[0] <= micro_bs:
                    print(
                        f"[Warn] OOM at epoch={epoch} step={step}, "
                        f"batch_size={x_opt.shape[0]} <= micro_batch_size={micro_bs}; skip batch."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    pred = None
                    loss = None
                    torch.cuda.empty_cache()
                    continue

                print(
                    f"[Warn] OOM at epoch={epoch} step={step}. "
                    f"Retry with micro-batch size={micro_bs}."
                )
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                n = int(x_opt.shape[0])
                recovered_loss = 0.0
                recovered_ok = True

                for start in range(0, n, micro_bs):
                    end = min(start + micro_bs, n)
                    x_opt_mb = x_opt[start:end]
                    x_sar_mb = x_sar[start:end]
                    y_mb = y[start:end]
                    pred_mb = None
                    loss_mb = None

                    try:
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            pred_mb = model(x_opt_mb, x_sar_mb)
                            loss_mb = criterion(pred_mb, y_mb)
                    except torch.cuda.OutOfMemoryError:
                        recovered_ok = False
                        break

                    loss_mb_value = float(loss_mb.detach().item())
                    if not torch.isfinite(loss_mb):
                        print(
                            f"[Warn] non-finite micro-batch loss at epoch={epoch} step={step}, "
                            f"micro=({start}:{end}), skip batch."
                        )
                        recovered_ok = False
                        break
                    if skip_loss_if_gt > 0 and loss_mb_value > skip_loss_if_gt:
                        print(
                            f"[Warn] abnormal micro-batch loss={loss_mb_value:.4f} > {skip_loss_if_gt:.4f} "
                            f"at epoch={epoch} step={step}, micro=({start}:{end}), skip batch."
                        )
                        recovered_ok = False
                        break

                    weight = float(end - start) / float(n)
                    scaler.scale(loss_mb * weight).backward()
                    recovered_loss += loss_mb_value * weight
                    pred_mb = None
                    loss_mb = None

                if not recovered_ok:
                    optimizer.zero_grad(set_to_none=True)
                    pred = None
                    loss = None
                    torch.cuda.empty_cache()
                    print(f"[Warn] OOM recovery failed at epoch={epoch} step={step}, skip this batch.")
                    continue

                loss_value = recovered_loss
                pred = None
                loss = None

            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            try:
                scaler.step(optimizer)
                scaler.update()
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[Warn] OOM during optimizer step at epoch={epoch} step={step}, skip this batch.")
                continue

            scheduler.step(global_step)
            if ema is not None:
                ema.update(model)

            total_loss += loss_value
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                avg=f"{total_loss / step:.4f}",
                lr=f"{optimizer.param_groups[-1]['lr']:.2e}",
            )

        train_loss = total_loss / max(len(train_loader), 1)
        val_scores = evaluate(model, val_loader, device, cfg, criterion, ema=ema, epoch=epoch)

        msg = (
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_scores['val_loss']:.4f} | "
            f"mIoU: {val_scores['mIoU']:.4f} | "
            f"OA: {val_scores['OA']:.4f} | "
            f"Kappa: {val_scores['Kappa']:.4f} | "
            f"mF1: {val_scores['mF1']:.4f}"
        )
        if "mIoU_all" in val_scores:
            msg += f" | mIoU_all: {val_scores['mIoU_all']:.4f}"
        if "mF1_all" in val_scores:
            msg += f" | mF1_all: {val_scores['mF1_all']:.4f}"
        print(msg)

        improved = val_scores["mIoU"] > (best_miou + early_stop_min_delta)
        should_early_stop = False
        if improved:
            best_miou = val_scores["mIoU"]
            no_improve_epochs = 0
            safe_torch_save(build_checkpoint_payload(model, ema, optimizer, epoch, best_miou, cfg), save_path)
            print(f"Saved best model to {save_path}, best mIoU={best_miou:.4f}")
        else:
            if epoch >= early_stop_start_epoch:
                no_improve_epochs += 1
                if early_stop_patience > 0:
                    print(f"No improvement for {no_improve_epochs}/{early_stop_patience} epochs.")
                    if no_improve_epochs >= early_stop_patience:
                        should_early_stop = True
            else:
                print(
                    f"No improvement at epoch {epoch}, "
                    f"but early-stop counting starts at epoch {early_stop_start_epoch}."
                )

        if bool(getattr(cfg, "save_last_checkpoint", True)):
            safe_torch_save(build_checkpoint_payload(model, ema, optimizer, epoch, best_miou, cfg), last_save_path)
        if should_early_stop:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
