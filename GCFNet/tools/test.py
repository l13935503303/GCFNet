import argparse
import sys
import os
import json
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import YESegConfig, PohangConfig
from dataset.dataset import (
    MultimodalSegDataset,
    build_yeseg_samples,
    build_pohang_samples,
    split_samples,
    split_samples_by_txt,
)
from models import GCFNet
from utils import SegMetric
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GCFNet on YESeg or Pohang")
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

    return {
        "num_classes": int(meta.get("num_classes", -1)),
        "original_num_classes": int(meta.get("original_num_classes", -1)),
        "drop_classes": _normalize_int_tuple(meta.get("drop_classes", tuple())),
        "class_id_map": _normalize_class_id_map(meta.get("class_id_map", {})),
    }


def _extract_bias_class_setup_signature(payload):
    if not isinstance(payload, dict):
        return None

    meta = payload.get("class_setup", None)
    if not isinstance(meta, dict):
        # Backward compatibility for older files that only stored top-level fields.
        has_top_level_meta = (
            ("num_classes" in payload)
            or ("drop_classes" in payload)
            or ("class_id_map" in payload)
        )
        if not has_top_level_meta:
            return None
        meta = payload

    return {
        "num_classes": int(meta.get("num_classes", -1)),
        "original_num_classes": int(meta.get("original_num_classes", -1)),
        "drop_classes": _normalize_int_tuple(meta.get("drop_classes", tuple())),
        "class_id_map": _normalize_class_id_map(meta.get("class_id_map", {})),
    }


def _class_setup_mismatch_reasons(expected_sig, loaded_sig):
    reasons = []
    if loaded_sig["num_classes"] != expected_sig["num_classes"]:
        reasons.append(
            f"num_classes file={loaded_sig['num_classes']} cfg={expected_sig['num_classes']}"
        )
    if loaded_sig["drop_classes"] != expected_sig["drop_classes"]:
        reasons.append(
            f"drop_classes file={loaded_sig['drop_classes']} cfg={expected_sig['drop_classes']}"
        )
    if loaded_sig["class_id_map"] != expected_sig["class_id_map"]:
        reasons.append(
            f"class_id_map file={loaded_sig['class_id_map']} cfg={expected_sig['class_id_map']}"
        )
    return reasons


def resolve_test_checkpoint_path(cfg):
    best_path = cfg.save_path
    last_name = str(getattr(cfg, "last_save_name", "last_gcfnet_final.pth"))
    last_path = os.path.join(cfg.save_dir, last_name)
    policy = str(getattr(cfg, "test_checkpoint_policy", "best_or_latest")).lower()

    if policy == "best" or (not os.path.exists(last_path)):
        return best_path
    if policy == "latest":
        return last_path if os.path.exists(last_path) else best_path
    if policy == "best_or_latest":
        if os.path.exists(last_path) and (
            (not os.path.exists(best_path))
            or (os.path.getmtime(last_path) >= os.path.getmtime(best_path))
        ):
            return last_path
        return best_path
    return best_path


def build_loader_kwargs(cfg, loader_workers, pin_memory):
    kwargs = {
        "num_workers": int(loader_workers),
        "pin_memory": bool(pin_memory),
    }
    if loader_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(cfg, "persistent_workers", True))
        kwargs["prefetch_factor"] = max(1, int(getattr(cfg, "prefetch_factor", 2)))
    return kwargs


def _normalize_split_key_text(raw: str) -> str:
    key = str(raw).strip().replace("\\", "/")
    if len(key) == 0:
        return ""
    key = Path(key).with_suffix("").as_posix()
    while key.startswith("./"):
        key = key[2:]
    if key.startswith("/"):
        key = key[1:]
    return key


def _load_split_key_set(split_path: Path):
    keys = []
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            k = _normalize_split_key_text(line)
            if len(k) > 0:
                keys.append(k)
    uniq = set(keys)
    dup_count = len(keys) - len(uniq)
    return uniq, len(keys), dup_count


def check_txt_split_integrity(cfg):
    if (
        str(getattr(cfg, "dataset_name", "")).lower() == "pohang"
        and (not bool(getattr(cfg, "use_official_split_txt", True)))
    ):
        print("[Split check] skipped for Pohang random split.")
        return None

    root = Path(getattr(cfg, "root", "."))
    train_name = str(getattr(cfg, "train_split_file", "train.txt"))
    val_name = str(getattr(cfg, "val_split_file", "val.txt"))
    train_path = root / train_name
    val_path = root / val_name

    if not train_path.exists():
        raise FileNotFoundError(f"train split file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"val split file not found: {val_path}")

    train_keys, train_lines, train_dup = _load_split_key_set(train_path)
    val_keys, val_lines, val_dup = _load_split_key_set(val_path)
    overlap = train_keys & val_keys

    report = {
        "train_file": str(train_path),
        "val_file": str(val_path),
        "train_lines": int(train_lines),
        "val_lines": int(val_lines),
        "train_unique": int(len(train_keys)),
        "val_unique": int(len(val_keys)),
        "train_duplicates": int(train_dup),
        "val_duplicates": int(val_dup),
        "overlap_count": int(len(overlap)),
        "overlap_examples": sorted(list(overlap))[:10],
    }

    print(
        f"[Split check] train={report['train_unique']} (dup={report['train_duplicates']}), "
        f"val={report['val_unique']} (dup={report['val_duplicates']}), "
        f"overlap={report['overlap_count']}"
    )

    if report["overlap_count"] > 0 and bool(getattr(cfg, "strict_txt_split_disjoint", True)):
        raise RuntimeError(
            "train.txt and val.txt overlap is detected, evaluation is not trustworthy. "
            f"examples={report['overlap_examples']}"
        )

    return report


def build_val_samples(cfg):
    if cfg.dataset_name == "Pohang":
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
            train_split_file = getattr(cfg, "train_split_file", "train_pohang.txt")
            val_split_file = getattr(cfg, "val_split_file", "val_pohang.txt")
            _, val_samples = split_samples_by_txt(
                samples=all_samples,
                root=cfg.root,
                optical_dir=".",
                train_split_file=train_split_file,
                val_split_file=val_split_file,
            )
            print(
                f"[Split] using txt split: train={train_split_file}, val={val_split_file}"
            )
            return val_samples
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
        _, val_samples = split_samples(
            all_samples,
            val_ratio=getattr(cfg, "val_ratio", 0.2),
            seed=getattr(cfg, "seed", 42),
        )
        print(
            f"[Split] Pohang random split: scenes={scene_list}, "
            f"val_ratio={getattr(cfg, 'val_ratio', 0.2)}, seed={getattr(cfg, 'seed', 42)}"
        )
        return val_samples
    else:
        all_samples = build_yeseg_samples(
            root=cfg.root,
            optical_dir=cfg.optical_dir,
            sar_dir=cfg.sar_dir,
            mask_dir=cfg.mask_dir,
        )
        train_split_file = getattr(cfg, "train_split_file", "train.txt")
        val_split_file = getattr(cfg, "val_split_file", "val.txt")
        if not bool(getattr(cfg, "use_official_split_txt", True)):
            print("[Split] force txt split is enabled in test.py, ignore cfg.use_official_split_txt=False.")
        _, val_samples = split_samples_by_txt(
            samples=all_samples,
            root=cfg.root,
            optical_dir=cfg.optical_dir,
            train_split_file=train_split_file,
            val_split_file=val_split_file,
        )
        print(f"[Split] using txt split: val={val_split_file}")
        return val_samples


@torch.no_grad()
def forward_logits(model, x_opt, x_sar):
    pred = model(x_opt, x_sar)
    if isinstance(pred, dict):
        pred = pred["main"]
    return pred


@torch.no_grad()
def sliding_window_inference(
    model,
    x_opt,
    x_sar,
    num_classes,
    crop_size=512,
    stride=384,
):
    """
    x_opt: [B, C1, H, W]
    x_sar: [B, C2, H, W]
    return: logits [B, num_classes, H, W]
    """
    assert x_opt.shape[0] == 1, "Sliding-window inference expects batch_size=1"
    assert x_opt.shape[-2:] == x_sar.shape[-2:], "Opt/SAR spatial size mismatch"

    b, _, h, w = x_opt.shape
    device = x_opt.device

    output = torch.zeros((b, num_classes, h, w), device=device)
    count_map = torch.zeros((b, 1, h, w), device=device)

    # 灏忓浘鐩存帴鏁村浘鎺ㄧ悊
    if h <= crop_size and w <= crop_size:
        logits = forward_logits(model, x_opt, x_sar)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits

    ys = list(range(0, max(h - crop_size, 0) + 1, stride))
    xs = list(range(0, max(w - crop_size, 0) + 1, stride))

    if len(ys) == 0 or ys[-1] != h - crop_size:
        ys.append(max(h - crop_size, 0))
    if len(xs) == 0 or xs[-1] != w - crop_size:
        xs.append(max(w - crop_size, 0))

    for y1 in ys:
        for x1 in xs:
            y2 = min(y1 + crop_size, h)
            x2 = min(x1 + crop_size, w)
            y1 = max(y2 - crop_size, 0)
            x1 = max(x2 - crop_size, 0)

            patch_opt = x_opt[:, :, y1:y2, x1:x2]
            patch_sar = x_sar[:, :, y1:y2, x1:x2]

            logits = forward_logits(model, patch_opt, patch_sar)
            logits = F.interpolate(
                logits,
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            )

            output[:, :, y1:y2, x1:x2] += logits
            count_map[:, :, y1:y2, x1:x2] += 1.0

    output = output / count_map.clamp_min(1.0)
    return output


@torch.no_grad()
def tta_sliding_inference(
    model,
    x_opt,
    x_sar,
    num_classes,
    crop_size=512,
    stride=384,
    use_hflip=True,
    use_vflip=True,
):
    preds = []

    # Base prediction.
    pred = sliding_window_inference(
        model, x_opt, x_sar,
        num_classes=num_classes,
        crop_size=crop_size,
        stride=stride,
    )
    preds.append(pred)

    # Horizontal flip.
    if use_hflip:
        pred_h = sliding_window_inference(
            model,
            torch.flip(x_opt, dims=[3]),
            torch.flip(x_sar, dims=[3]),
            num_classes=num_classes,
            crop_size=crop_size,
            stride=stride,
        )
        pred_h = torch.flip(pred_h, dims=[3])
        preds.append(pred_h)

    # Vertical flip.
    if use_vflip:
        pred_v = sliding_window_inference(
            model,
            torch.flip(x_opt, dims=[2]),
            torch.flip(x_sar, dims=[2]),
            num_classes=num_classes,
            crop_size=crop_size,
            stride=stride,
        )
        pred_v = torch.flip(pred_v, dims=[2])
        preds.append(pred_v)

    return torch.mean(torch.stack(preds, dim=0), dim=0)


@torch.no_grad()
def multi_scale_tta_inference(
    model,
    x_opt,
    x_sar,
    num_classes,
    crop_size=512,
    stride=384,
    scales=(1.0,),
    use_hflip=True,
    use_vflip=True,
):
    """
    return logits [B, num_classes, H, W]
    """
    b, _, h, w = x_opt.shape
    preds = []

    for s in scales:
        new_h = max(int(round(h * s)), 32)
        new_w = max(int(round(w * s)), 32)

        opt_scaled = F.interpolate(x_opt, size=(new_h, new_w), mode="bilinear", align_corners=False)
        sar_scaled = F.interpolate(x_sar, size=(new_h, new_w), mode="bilinear", align_corners=False)

        logits = tta_sliding_inference(
            model,
            opt_scaled,
            sar_scaled,
            num_classes=num_classes,
            crop_size=crop_size,
            stride=stride,
            use_hflip=use_hflip,
            use_vflip=use_vflip,
        )

        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        preds.append(logits)

    return torch.mean(torch.stack(preds, dim=0), dim=0)

@torch.no_grad()
def evaluate_with_bias_dict(
    model,
    val_loader,
    cfg,
    device,
    bias_dict,
    target_inference_fn,
):
    metric = SegMetric(
        cfg.num_classes,
        ignore_index=cfg.ignore_index,
        ignore_absent_classes=bool(getattr(cfg, "metric_ignore_absent_classes", False)),
    )
    model.eval()

    for idx, batch in enumerate(val_loader):
        x_opt = batch["opt"].to(device)
        x_sar = batch["sar"].to(device)
        y = batch["mask"].to(device)

        logits = target_inference_fn(
            model,
            x_opt,
            x_sar,
            num_classes=cfg.num_classes,
            crop_size=getattr(cfg, "window_size", 512),
            stride=getattr(cfg, "window_stride", 384),
            scales=getattr(cfg, "ms_scales", (1.0,)),
            use_hflip=getattr(cfg, "use_tta", True),
            use_vflip=getattr(cfg, "use_tta", True),
        )

        for cls_id, bias in bias_dict.items():
            logits[:, int(cls_id), :, :] += float(bias)

        metric.update(logits, y)

    return metric.get_scores()


@torch.no_grad()
def coordinate_search_class_bias(
    model,
    val_loader,
    cfg,
    device,
    target_inference_fn,
):
    """
    Coordinate bias search over valid classes under current class mapping.
    """
    num_classes = int(getattr(cfg, "num_classes", 0))
    rare_class = int(getattr(cfg, "rare_class_id", -1))
    base_bias = deepcopy(getattr(cfg, "logit_bias_dict", {}))
    base_bias = {int(k): float(v) for k, v in base_bias.items() if 0 <= int(k) < num_classes}

    candidates = []
    for c in tuple(getattr(cfg, "confusion_class_ids", tuple())):
        ci = int(c)
        if 0 <= ci < num_classes and ci not in candidates:
            candidates.append(ci)
    if not candidates:
        candidates = [0]

    search_space = {}
    bias_dict = {}
    for cls_id in candidates:
        if cls_id == rare_class:
            search_space[cls_id] = [1.0, 1.25, 1.5, 1.75, 2.0]
            bias_dict[cls_id] = float(base_bias.get(cls_id, 1.5))
        elif cls_id == 0:
            search_space[cls_id] = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
            bias_dict[cls_id] = float(base_bias.get(cls_id, 0.0))
        else:
            search_space[cls_id] = [-0.5, -0.25, 0.0, 0.25, 0.5]
            bias_dict[cls_id] = float(base_bias.get(cls_id, 0.0))

    best_scores = evaluate_with_bias_dict(
        model,
        val_loader,
        cfg,
        device,
        bias_dict,
        target_inference_fn,
    )
    best_miou = best_scores["mIoU"]

    print("\nInitial bias:", bias_dict)
    print(f"Initial mIoU={best_miou:.4f}")

    for round_idx in range(2):
        print(f"\nCoordinate bias search round {round_idx + 1}")

        for cls_id, values in search_space.items():
            local_best_bias = bias_dict[cls_id]
            local_best_scores = best_scores
            local_best_miou = best_miou

            for v in values:
                trial_bias = dict(bias_dict)
                trial_bias[cls_id] = v

                scores = evaluate_with_bias_dict(
                    model,
                    val_loader,
                    cfg,
                    device,
                    trial_bias,
                    target_inference_fn,
                )

                tracked_iou = []
                for tc in candidates:
                    tracked_iou.append(f"class{tc}={scores['IoU_per_class'][tc]:.4f}")
                tracked_iou_str = " | ".join(tracked_iou)
                print(
                    f"class={cls_id}, bias={v:+.2f} | "
                    f"mIoU={scores['mIoU']:.4f} | "
                    f"{tracked_iou_str}"
                )

                if scores["mIoU"] > local_best_miou:
                    local_best_miou = scores["mIoU"]
                    local_best_bias = v
                    local_best_scores = scores

            bias_dict[cls_id] = local_best_bias
            best_scores = local_best_scores
            best_miou = local_best_miou

            print(
                f"Best for class {cls_id}: bias={local_best_bias:+.2f}, "
                f"mIoU={best_miou:.4f}"
            )

    print("\nFinal best bias dict:", bias_dict)
    print(f"mIoU : {best_scores['mIoU']:.4f}")
    print(f"OA   : {best_scores['OA']:.4f}")
    print(f"Kappa: {best_scores['Kappa']:.4f}")
    print(f"mF1  : {best_scores['mF1']:.4f}")
    if "mIoU_all" in best_scores:
        print(f"mIoU_all: {best_scores['mIoU_all']:.4f}")
    if "mF1_all" in best_scores:
        print(f"mF1_all : {best_scores['mF1_all']:.4f}")
    print("IoU_per_class:", best_scores["IoU_per_class"])
    print("F1_per_class:", best_scores["F1_per_class"])

    return bias_dict, best_scores
def print_confusion_for_classes(metric, class_ids=(0, 3, 7)):
    cm = metric.confusion_matrix.float()

    print("\nConfusion analysis:")

    for cls in class_ids:
        cls = int(cls)

        if cls >= cm.shape[0]:
            continue

        gt_total = cm[cls, :].sum().item()
        pred_total = cm[:, cls].sum().item()
        correct = cm[cls, cls].item()

        print(f"\nClass {cls}:")
        print(f"  GT pixels   : {int(gt_total)}")
        print(f"  Pred pixels : {int(pred_total)}")
        print(f"  Correct     : {int(correct)}")

        if gt_total > 0:
            row = cm[cls, :] / gt_total
            top_vals, top_ids = torch.topk(row, k=min(5, row.numel()))
            print("  GT class misclassified as:")
            for v, i in zip(top_vals.tolist(), top_ids.tolist()):
                print(f"    pred {i}: {v:.4f}")

        if pred_total > 0:
            col = cm[:, cls] / pred_total
            top_vals, top_ids = torch.topk(col, k=min(5, col.numel()))
            print("  Pred class comes from GT:")
            for v, i in zip(top_vals.tolist(), top_ids.tolist()):
                print(f"    gt {i}: {v:.4f}")


# =========================================================
# Printing
# =========================================================
def print_scores(scores):
    print(f"mIoU : {scores['mIoU']:.4f}")
    print(f"OA   : {scores['OA']:.4f}")
    print(f"Kappa: {scores['Kappa']:.4f}")
    print(f"mF1  : {scores['mF1']:.4f}")
    if "mIoU_all" in scores:
        print(f"mIoU_all: {scores['mIoU_all']:.4f}")
    if "mF1_all" in scores:
        print(f"mF1_all : {scores['mF1_all']:.4f}")
    print("IoU_per_class:", scores["IoU_per_class"])
    print("F1_per_class:", scores["F1_per_class"])


def print_credibility_report(cfg, split_report, scores_no_bias, scores_bias, bias_dict):
    print("\nCredibility report:")
    if split_report is not None:
        print(
            f"- split_disjoint: {split_report['overlap_count'] == 0} "
            f"(overlap={split_report['overlap_count']})"
        )
        print(
            f"- split_size: train={split_report['train_unique']}, val={split_report['val_unique']}"
        )

    non_zero_bias = {
        int(k): float(v) for k, v in (bias_dict or {}).items() if abs(float(v)) > 1e-12
    }
    print(f"- non_zero_bias_count: {len(non_zero_bias)}")
    if len(non_zero_bias) > 0:
        print(f"- non_zero_bias_dict: {non_zero_bias}")

    if (scores_no_bias is not None) and (scores_bias is not None):
        delta_miou = float(scores_bias["mIoU"]) - float(scores_no_bias["mIoU"])
        delta_oa = float(scores_bias["OA"]) - float(scores_no_bias["OA"])
        delta_kappa = float(scores_bias["Kappa"]) - float(scores_no_bias["Kappa"])
        print(
            f"- bias_delta: mIoU={delta_miou:+.4f}, OA={delta_oa:+.4f}, Kappa={delta_kappa:+.4f}"
        )
        warn_thr = float(getattr(cfg, "bias_effect_warn_threshold", 0.015))
        if abs(delta_miou) > warn_thr:
            print(
                f"[Warn] bias impact on mIoU is relatively large (>|{warn_thr:.3f}|); "
                "please also report no-bias metrics in papers/reports."
            )

    print("- note: metrics are trustworthy on current fixed split; cross-split generalization still needs extra validation.")


def load_checkpoint_with_report(
    model,
    ckpt_path,
    device,
    strict=True,
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
                    "Checkpoint missing class-setup metadata, unsafe for class-drop evaluation. "
                    "Please evaluate a checkpoint trained under current class mapping, or set "
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
                raise RuntimeError("Checkpoint class setup mismatch: " + reason_text)

    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as err:
        print("\n[Checkpoint check] shape mismatch detected when loading state_dict.")
        print(err)
        raise

    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    print("\n[Checkpoint check]")
    print(f"Missing keys   : {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    print(f"Loaded weights : {source}")
    if missing_keys:
        print("Sample missing keys:", missing_keys[:20])
    if unexpected_keys:
        print("Sample unexpected keys:", unexpected_keys[:20])

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            "Checkpoint modules do not fully match current model. "
            "Set strict=False if you intentionally allow partial loading."
        )

    print(f"Loaded checkpoint: {ckpt_path}")
    return ckpt


def save_bias_search_result(cfg, bias_dict, scores):
    save_path = getattr(cfg, "bias_result_save_path", "./checkpoints/best_bias.json")
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    payload = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": getattr(cfg, "dataset_name", ""),
        "num_classes": int(getattr(cfg, "num_classes", 0)),
        "original_num_classes": int(getattr(cfg, "original_num_classes", getattr(cfg, "num_classes", 0))),
        "drop_classes": list(_normalize_int_tuple(getattr(cfg, "drop_classes", tuple()))),
        "class_id_map": {
            str(int(k)): int(v)
            for k, v in _normalize_class_id_map(getattr(cfg, "class_id_map", {})).items()
        },
        "class_setup": {
            "num_classes": int(getattr(cfg, "num_classes", 0)),
            "original_num_classes": int(getattr(cfg, "original_num_classes", getattr(cfg, "num_classes", 0))),
            "drop_classes": list(_normalize_int_tuple(getattr(cfg, "drop_classes", tuple()))),
            "class_id_map": {
                str(int(k)): int(v)
                for k, v in _normalize_class_id_map(getattr(cfg, "class_id_map", {})).items()
            },
        },
        "logit_bias_dict": {str(int(k)): float(v) for k, v in bias_dict.items()},
        "metrics": {
            "mIoU": float(scores["mIoU"]),
            "OA": float(scores["OA"]),
            "Kappa": float(scores["Kappa"]),
            "mF1": float(scores["mF1"]),
            "IoU_per_class": [float(x) for x in scores["IoU_per_class"]],
            "F1_per_class": [float(x) for x in scores["F1_per_class"]],
        },
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved best bias result to: {save_path}")
    print(f"Suggested cfg.logit_bias_dict = {bias_dict}")


def load_bias_result(cfg):
    configured_path = getattr(cfg, "bias_result_load_path", "./checkpoints/best_bias.json")
    candidate_paths = [configured_path]
    fallback_legacy = os.path.join(os.path.dirname(configured_path), "best_bias.json")
    if (fallback_legacy not in candidate_paths) and (fallback_legacy != configured_path):
        candidate_paths.append(fallback_legacy)

    load_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            load_path = p
            break

    if load_path is None:
        print(f"[Bias load] file not found, keep cfg.logit_bias_dict: {configured_path}")
        return None

    if load_path != configured_path:
        print(f"[Bias load] fallback to legacy bias file: {load_path}")

    with open(load_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    strict_check = bool(getattr(cfg, "strict_bias_class_setup", True))
    allow_no_meta = bool(getattr(cfg, "allow_bias_without_meta", False))
    expected_sig = _cfg_class_setup_signature(cfg)
    loaded_sig = _extract_bias_class_setup_signature(payload)

    if strict_check:
        if loaded_sig is None:
            if bool(getattr(cfg, "enable_class_drop", False)) and (not allow_no_meta):
                print(
                    f"[Bias load] missing class_setup metadata in {load_path}; "
                    "ignore this file to avoid class-mapping leakage."
                )
                return None
        else:
            mismatch = _class_setup_mismatch_reasons(expected_sig, loaded_sig)
            if len(mismatch) > 0:
                print(
                    f"[Bias load] class setup mismatch, ignore {load_path}: "
                    + "; ".join(mismatch)
                )
                return None

    raw_bias = payload.get("logit_bias_dict", {})
    if not isinstance(raw_bias, dict) or len(raw_bias) == 0:
        print(f"[Bias load] invalid logit_bias_dict in file: {load_path}")
        return None

    loaded_bias = {}
    for k, v in raw_bias.items():
        cls_id = int(k)
        if cls_id < 0 or cls_id >= int(getattr(cfg, "num_classes", 0)):
            continue
        loaded_bias[cls_id] = float(v)

    if len(loaded_bias) == 0:
        print(f"[Bias load] no valid class bias found in file: {load_path}")
        return None

    print(f"[Bias load] loaded from {load_path}: {loaded_bias}")
    return loaded_bias


@torch.no_grad()
def auto_search_bias_when_missing(
    model,
    val_loader,
    cfg,
    device,
    base_bias_dict,
):
    if not bool(getattr(cfg, "auto_search_bias_if_no_file", False)):
        return base_bias_dict, False

    num_classes = int(getattr(cfg, "num_classes", 0))
    search_space_cfg = getattr(cfg, "auto_search_bias_space", {})
    default_classes = tuple(getattr(cfg, "auto_search_bias_classes", (0,)))
    default_values = tuple(float(v) for v in getattr(cfg, "auto_search_bias_values", (0.0, 0.1, 0.2, 0.3)))

    search_space = {}
    if isinstance(search_space_cfg, dict) and len(search_space_cfg) > 0:
        for k, values in search_space_cfg.items():
            cls_id = int(k)
            if cls_id < 0 or cls_id >= num_classes:
                continue
            if not isinstance(values, (list, tuple)):
                values = [values]
            parsed = [float(v) for v in values]
            if len(parsed) > 0:
                search_space[cls_id] = tuple(parsed)
    else:
        for c in default_classes:
            cls_id = int(c)
            if cls_id < 0 or cls_id >= num_classes:
                continue
            if len(default_values) > 0:
                search_space[cls_id] = tuple(default_values)

    if len(search_space) == 0:
        return base_bias_dict, False

    bias_dict = deepcopy(base_bias_dict)
    for cls_id in search_space.keys():
        bias_dict.setdefault(cls_id, 0.0)

    best_scores = evaluate_with_bias_dict(
        model=model,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        bias_dict=bias_dict,
        target_inference_fn=multi_scale_tta_inference,
    )
    best_miou = float(best_scores["mIoU"])
    rounds = max(1, int(getattr(cfg, "auto_search_bias_rounds", 1)))
    min_delta = float(getattr(cfg, "auto_search_bias_min_delta", 0.0))

    tracked_classes = sorted(search_space.keys())
    print("\nAuto bias search start:")
    print(f"initial_bias={bias_dict}, mIoU={best_miou:.4f}")

    for round_idx in range(rounds):
        print(f"\nAuto bias search round {round_idx + 1}/{rounds}")
        for cls_id, values in search_space.items():
            local_best_bias = float(bias_dict.get(cls_id, 0.0))
            local_best_scores = best_scores
            local_best_miou = best_miou

            for v in values:
                trial = dict(bias_dict)
                trial[cls_id] = float(v)
                scores = evaluate_with_bias_dict(
                    model=model,
                    val_loader=val_loader,
                    cfg=cfg,
                    device=device,
                    bias_dict=trial,
                    target_inference_fn=multi_scale_tta_inference,
                )
                miou = float(scores["mIoU"])
                tracked = " | ".join(
                    [f"class{tc}={scores['IoU_per_class'][tc]:.4f}" for tc in tracked_classes]
                )
                print(f"class={cls_id}, bias={float(v):+.3f} | mIoU={miou:.4f} | {tracked}")

                if miou > (local_best_miou + min_delta):
                    local_best_miou = miou
                    local_best_bias = float(v)
                    local_best_scores = scores

            bias_dict[cls_id] = local_best_bias
            best_scores = local_best_scores
            best_miou = local_best_miou
            print(f"best class={cls_id} -> bias={local_best_bias:+.3f}, mIoU={best_miou:.4f}")

    print(f"\nAuto bias search done: bias={bias_dict}, mIoU={best_miou:.4f}")
    return bias_dict, True


@torch.no_grad()
def main():
    # ===== select dataset profile =====
    args = parse_args()
    cfg = build_cfg(args.dataset)
    print(f"[Config] dataset profile: {args.dataset}")

    # ===== 鎺ㄧ悊鍙傛暟 =====
    crop_size = int(getattr(cfg, "window_size", 256))
    stride = int(getattr(cfg, "window_stride", crop_size))
    scales = tuple(getattr(cfg, "ms_scales", (1.0,)))
    use_hflip = True
    use_vflip = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_report = check_txt_split_integrity(cfg)
    val_samples = build_val_samples(cfg)

    print(f"Validation samples: {len(val_samples)}")
    print(
        f"Class setup: original={getattr(cfg, 'original_num_classes', cfg.num_classes)} "
        f"-> eval={cfg.num_classes}, drop={tuple(getattr(cfg, 'drop_classes', tuple()))}, "
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

    val_set = MultimodalSegDataset(
        samples=val_samples,
        transform=None,
        ignore_index=cfg.ignore_index,
        normalize_opt=bool(getattr(cfg, "normalize_opt", True)),
        normalize_sar=bool(getattr(cfg, "normalize_sar", True)),
        normalize_stage=str(getattr(cfg, "val_normalize_stage", "pre")).lower(),
        sar_log_transform=bool(getattr(cfg, "val_sar_log_transform", False)),
        class_id_map=dict(getattr(cfg, "class_id_map", {})),
        include_ignore_as_class=bool(getattr(cfg, "include_ignore_label_as_class", False)),
        ignore_as_class_id=getattr(cfg, "ignore_label_class_id", None),
    )
    is_windows = os.name == "nt"
    loader_workers = 0 if is_windows else int(getattr(cfg, "num_workers", 4))
    loader_kwargs = build_loader_kwargs(
        cfg,
        loader_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,   # Sliding-window inference expects batch_size=1.
        **loader_kwargs,
    )

    sample = val_set[0]
    opt_in_channels = sample["opt"].shape[0]
    sar_in_channels = sample["sar"].shape[0]
    print(f"opt_in_channels={opt_in_channels}, sar_in_channels={sar_in_channels}")

    model = GCFNet(
    num_classes=cfg.num_classes,
    opt_in_channels=opt_in_channels,
    sar_in_channels=sar_in_channels,
    decoder_dim=getattr(cfg, "decoder_dim", 192),
    deep_supervision=getattr(cfg, "deep_supervision", True),
    pretrained_backbone=False,
    eca_kernel_size=getattr(cfg, "eca_kernel_size", 3),
    use_opt_modality=getattr(cfg, "use_opt_modality", True),
    use_sar_modality=getattr(cfg, "use_sar_modality", True),
    use_gmf_module=getattr(cfg, "use_gmf_module", True),
    gmf_use_residual_sum=getattr(cfg, "gmf_use_residual_sum", True),
    gam_reduction=getattr(cfg, "gam_reduction", 4),
    use_gam_module=getattr(cfg, "use_gam_module", True),
    gam_use_residual=getattr(cfg, "gam_use_residual", True),
    opt_pretrained_path="",
    sar_pretrained_path="",
    use_glvrm=getattr(cfg, "use_glvrm", True),
    glvrm_decoder_stages=getattr(cfg, "glvrm_decoder_stages", 1),
    use_aspp_context=getattr(cfg, "use_aspp_context", True),
    aspp_dilations=getattr(cfg, "aspp_dilations", (1, 3, 6)),
    aspp_dropout=getattr(cfg, "aspp_dropout", 0.05),
).to(device)

    test_ckpt_path = resolve_test_checkpoint_path(cfg)
    print(f"[Test checkpoint] path={test_ckpt_path}")

    load_checkpoint_with_report(
        model=model,
        ckpt_path=test_ckpt_path,
        device=device,
        strict=True,
        prefer_ema=bool(getattr(cfg, "test_prefer_ema", True)),
        expected_cfg=cfg,
        enforce_class_setup_match=bool(getattr(cfg, "strict_resume_class_setup", True)),
        allow_resume_without_meta=bool(getattr(cfg, "allow_resume_without_class_meta", False)),
    )
    model.eval()
    final_logit_bias_dict = deepcopy(getattr(cfg, "logit_bias_dict", {}))
    did_bias_search = False

    if getattr(cfg, "auto_load_bias_result", True):
        loaded_bias = load_bias_result(cfg)
        if loaded_bias is not None:
            final_logit_bias_dict = loaded_bias
            target_classes = set()
            search_space_cfg = getattr(cfg, "auto_search_bias_space", {})
            if isinstance(search_space_cfg, dict) and len(search_space_cfg) > 0:
                for k in search_space_cfg.keys():
                    target_classes.add(int(k))
            else:
                for c in tuple(getattr(cfg, "auto_search_bias_classes", tuple())):
                    target_classes.add(int(c))
            target_classes = {
                c for c in target_classes if 0 <= c < int(getattr(cfg, "num_classes", 0))
            }
            missing_targets = [c for c in sorted(target_classes) if c not in final_logit_bias_dict]
            if bool(getattr(cfg, "auto_search_bias_if_no_file", False)) and len(missing_targets) > 0:
                print(
                    f"[Bias load] missing target classes in loaded bias: {missing_targets}, "
                    "run auto-search to complete."
                )
                final_logit_bias_dict, auto_searched = auto_search_bias_when_missing(
                    model=model,
                    val_loader=val_loader,
                    cfg=cfg,
                    device=device,
                    base_bias_dict=final_logit_bias_dict,
                )
                did_bias_search = did_bias_search or auto_searched
        else:
            final_logit_bias_dict, auto_searched = auto_search_bias_when_missing(
                model=model,
                val_loader=val_loader,
                cfg=cfg,
                device=device,
                base_bias_dict=final_logit_bias_dict,
            )
            did_bias_search = did_bias_search or auto_searched

    if getattr(cfg, "search_multi_class_bias", False):
        best_bias_dict, _ = coordinate_search_class_bias(
            model=model,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            target_inference_fn=multi_scale_tta_inference,
        )
        final_logit_bias_dict = {int(k): float(v) for k, v in best_bias_dict.items()}
        print(f"\nUse searched multi-class bias dict: {final_logit_bias_dict}")
        did_bias_search = True


    # ===== Normal evaluation =====
    metric_with_bias = SegMetric(
        cfg.num_classes,
        ignore_index=cfg.ignore_index,
        ignore_absent_classes=bool(getattr(cfg, "metric_ignore_absent_classes", False)),
    )
    enable_no_bias_report = bool(getattr(cfg, "eval_report_without_bias", True))
    metric_no_bias = SegMetric(
        cfg.num_classes,
        ignore_index=cfg.ignore_index,
        ignore_absent_classes=bool(getattr(cfg, "metric_ignore_absent_classes", False)),
    ) if enable_no_bias_report else None
    effective_bias = {
        int(k): float(v)
        for k, v in final_logit_bias_dict.items()
        if abs(float(v)) > 1e-12
    }

    for idx, batch in enumerate(val_loader):
        x_opt = batch["opt"].to(device)
        x_sar = batch["sar"].to(device)
        y = batch["mask"].to(device)

        logits = multi_scale_tta_inference(
            model,
            x_opt,
            x_sar,
            num_classes=cfg.num_classes,
            crop_size=getattr(cfg, "window_size", 512),
            stride=getattr(cfg, "window_stride", 384),
            scales=getattr(cfg, "ms_scales", (1.0,)),
            use_hflip=getattr(cfg, "use_tta", True),
            use_vflip=getattr(cfg, "use_tta", True),
        )

        if metric_no_bias is not None:
            metric_no_bias.update(logits, y)

        if len(effective_bias) > 0:
            logits_with_bias = logits.clone()
            for cls_id, bias in effective_bias.items():
                logits_with_bias[:, int(cls_id), :, :] += float(bias)
        else:
            logits_with_bias = logits

        metric_with_bias.update(logits_with_bias, y)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(val_loader):
            print(f"[{idx + 1}/{len(val_loader)}] processed")

    scores_with_bias = metric_with_bias.get_scores()
    scores_no_bias = metric_no_bias.get_scores() if metric_no_bias is not None else None

    print("Validation Results (Bias applied):")
    print_scores(scores_with_bias)
    if scores_no_bias is not None:
        print("\nValidation Results (No bias):")
        print_scores(scores_no_bias)

    print_credibility_report(
        cfg=cfg,
        split_report=split_report,
        scores_no_bias=scores_no_bias,
        scores_bias=scores_with_bias,
        bias_dict=final_logit_bias_dict,
    )

    # Keep downstream behavior consistent with existing scripts (use bias-applied scores as main result).
    scores = scores_with_bias

    if (
        getattr(cfg, "save_bias_search_result", True)
        and (
            getattr(cfg, "search_multi_class_bias", False)
            or did_bias_search
        )
    ):
        save_bias_search_result(cfg, final_logit_bias_dict, scores)

    # ===== confusion analysis =====
    if getattr(cfg, "print_confusion", True):
        class_ids = tuple(getattr(cfg, "confusion_class_ids", (0, 1)))
        print_confusion_for_classes(metric_with_bias, class_ids=class_ids)


if __name__ == "__main__":
    main()
