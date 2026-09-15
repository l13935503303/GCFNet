import argparse
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import YESegConfig, PohangConfig
from models import GCFNet


def parse_args():
    parser = argparse.ArgumentParser(description="Profile Params/FLOPs/Latency for GCFNet.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.getenv("GCFNET_DATASET", "yeseg"),
        choices=["yeseg", "pohang"],
        help="Config profile to build model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Profiling device.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy input.")
    parser.add_argument("--height", type=int, default=256, help="Input height.")
    parser.add_argument("--width", type=int, default=256, help="Input width.")
    parser.add_argument("--opt-ch", type=int, default=3, help="Optical input channels.")
    parser.add_argument("--sar-ch", type=int, default=1, help="SAR input channels.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations for latency.")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations for latency.")
    parser.add_argument("--amp", action="store_true", help="Enable autocast during latency/FLOPs run.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path. Empty means auto-try cfg.save_path.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional output json path. Empty means auto-save under checkpoints/",
    )
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cfg(dataset_name: str):
    if str(dataset_name).strip().lower() == "pohang":
        return PohangConfig()
    return YESegConfig()


class MainOutputWrapper(nn.Module):
    """
    Make model output a single tensor for FLOPs tools that cannot handle dict outputs.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x_opt, x_sar):
        out = self.model(x_opt, x_sar)
        if isinstance(out, dict):
            if "main" in out and torch.is_tensor(out["main"]):
                return out["main"]
            for v in out.values():
                if torch.is_tensor(v):
                    return v
                if isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[0]):
                    return v[0]
            raise RuntimeError("Cannot find tensor output in model dict output.")
        return out


def build_model(cfg, opt_in_channels: int, sar_in_channels: int):
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
    )
    model.eval()
    return model


def select_state_dict(ckpt):
    source = "raw_checkpoint"
    state_dict = ckpt
    if isinstance(ckpt, dict):
        if ("model_ema" in ckpt) and (ckpt["model_ema"] is not None):
            state_dict = ckpt["model_ema"]
            source = "model_ema"
        elif ("model" in ckpt) and ckpt["model"] is not None:
            state_dict = ckpt["model"]
            source = "model"
        elif ("model_raw" in ckpt) and ckpt["model_raw"] is not None:
            state_dict = ckpt["model_raw"]
            source = "model_raw"
        elif ("ema" in ckpt) and isinstance(ckpt["ema"], dict) and len(ckpt["ema"]) > 0:
            state_dict = ckpt["ema"]
            source = "ema_shadow"
    return state_dict, source


def maybe_load_checkpoint(model, ckpt_path: str, device: torch.device):
    if len(str(ckpt_path).strip()) == 0:
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict, source = select_state_dict(ckpt)
    incompat = model.load_state_dict(state_dict, strict=False)
    print(
        f"[Checkpoint] loaded={ckpt_path} ({source}), "
        f"missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)}"
    )


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def try_flops_with_profiler(model, x_opt, x_sar, device, use_amp=False):
    if not hasattr(torch, "profiler"):
        return None, "torch.profiler unavailable"

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    wrapped = MainOutputWrapper(model).eval()
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=True,
        ) as prof:
            with torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda")):
                _ = wrapped(x_opt, x_sar)
        flops = 0
        for evt in prof.key_averages():
            val = getattr(evt, "flops", None)
            if val is not None:
                flops += int(val)
        if flops <= 0:
            return None, "torch.profiler returned zero FLOPs"
        return float(flops), "torch.profiler"
    except Exception as err:
        return None, f"torch.profiler failed: {err}"


@torch.no_grad()
def try_flops_with_thop(model, x_opt, x_sar):
    try:
        from thop import profile as thop_profile
    except Exception:
        return None, "thop unavailable"

    wrapped = MainOutputWrapper(model).eval()
    try:
        macs, _ = thop_profile(wrapped, inputs=(x_opt, x_sar), verbose=False)
        flops = float(macs) * 2.0
        return flops, "thop (MACs*2)"
    except Exception as err:
        return None, f"thop failed: {err}"


@torch.no_grad()
def try_flops_with_fvcore(model, x_opt, x_sar):
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return None, "fvcore unavailable"

    wrapped = MainOutputWrapper(model).eval()
    try:
        flops = float(FlopCountAnalysis(wrapped, (x_opt, x_sar)).total())
        return flops, "fvcore"
    except Exception as err:
        return None, f"fvcore failed: {err}"


def format_number(x):
    if x is None:
        return "N/A"
    return f"{x:,.2f}"


def human_params(n):
    if n >= 1e9:
        return f"{n / 1e9:.3f} B"
    if n >= 1e6:
        return f"{n / 1e6:.3f} M"
    if n >= 1e3:
        return f"{n / 1e3:.3f} K"
    return str(int(n))


def human_flops(n):
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"{n / 1e12:.3f} TFLOPs"
    if n >= 1e9:
        return f"{n / 1e9:.3f} GFLOPs"
    if n >= 1e6:
        return f"{n / 1e6:.3f} MFLOPs"
    return f"{n:.1f} FLOPs"


@torch.no_grad()
def measure_latency_ms(model, x_opt, x_sar, device, warmup=20, iters=100, use_amp=False):
    wrapped = MainOutputWrapper(model).eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    for _ in range(max(0, int(warmup))):
        with torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda")):
            _ = wrapped(x_opt, x_sar)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(max(1, int(iters))):
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda")):
            _ = wrapped(x_opt, x_sar)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "iters": int(len(times)),
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    cfg = build_cfg(args.dataset)
    model = build_model(cfg, opt_in_channels=args.opt_ch, sar_in_channels=args.sar_ch).to(device)
    ckpt_path = str(args.checkpoint).strip()
    if len(ckpt_path) == 0:
        auto_ckpt = str(getattr(cfg, "save_path", "")).strip()
        if len(auto_ckpt) > 0 and os.path.exists(auto_ckpt):
            ckpt_path = auto_ckpt
            print(f"[Checkpoint] auto-resolve to cfg.save_path: {ckpt_path}")
    maybe_load_checkpoint(model, ckpt_path, device)

    b = int(args.batch_size)
    h = int(args.height)
    w = int(args.width)
    x_opt = torch.randn(b, int(args.opt_ch), h, w, device=device)
    x_sar = torch.randn(b, int(args.sar_ch), h, w, device=device)

    total_params, trainable_params = count_params(model)

    flops, flops_source = try_flops_with_profiler(model, x_opt, x_sar, device, use_amp=args.amp)
    if flops is None:
        flops, flops_source = try_flops_with_thop(model, x_opt, x_sar)
    if flops is None:
        flops, flops_source = try_flops_with_fvcore(model, x_opt, x_sar)

    latency = measure_latency_ms(
        model=model,
        x_opt=x_opt,
        x_sar=x_sar,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        use_amp=args.amp,
    )

    throughput = (1000.0 * b) / max(latency["mean_ms"], 1e-9)
    flops_per_sample = None if flops is None else (flops / max(b, 1))

    print("\n=== Model Profile ===")
    print(f"Dataset profile : {args.dataset}")
    print(f"Device          : {device}")
    print(f"Input shape     : opt=({b},{args.opt_ch},{h},{w}), sar=({b},{args.sar_ch},{h},{w})")
    print(f"AMP             : {bool(args.amp)}")
    print(f"Params(total)   : {human_params(total_params)} ({format_number(total_params)})")
    print(f"Params(trainable): {human_params(trainable_params)} ({format_number(trainable_params)})")
    print(f"FLOPs/source    : {human_flops(flops)} ({flops_source})")
    if flops_per_sample is not None:
        print(f"FLOPs/sample    : {human_flops(flops_per_sample)}")
    print(
        "Latency(ms)     : "
        f"mean={latency['mean_ms']:.3f}, std={latency['std_ms']:.3f}, "
        f"p50={latency['p50_ms']:.3f}, p90={latency['p90_ms']:.3f}, p95={latency['p95_ms']:.3f}, "
        f"min={latency['min_ms']:.3f}, max={latency['max_ms']:.3f}"
    )
    print(f"Throughput      : {throughput:.3f} samples/s")
    print(f"Latency iters   : warmup={args.warmup}, measure={latency['iters']}")

    save_json = str(args.save_json).strip()
    if len(save_json) == 0:
        os.makedirs("checkpoints", exist_ok=True)
        save_json = os.path.join("checkpoints", f"profile_{args.dataset}.json")

    result = {
        "dataset": str(args.dataset),
        "device": str(device),
        "amp": bool(args.amp),
        "input_shape": {
            "batch_size": int(b),
            "opt_channels": int(args.opt_ch),
            "sar_channels": int(args.sar_ch),
            "height": int(h),
            "width": int(w),
        },
        "checkpoint": ckpt_path if len(ckpt_path) > 0 else None,
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "flops_total": None if flops is None else float(flops),
        "flops_per_sample": None if flops_per_sample is None else float(flops_per_sample),
        "flops_source": str(flops_source),
        "latency_ms": latency,
        "throughput_samples_per_s": float(throughput),
    }
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved profile json: {save_json}")


if __name__ == "__main__":
    main()
