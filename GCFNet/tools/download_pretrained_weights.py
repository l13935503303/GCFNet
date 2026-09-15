import importlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "pretrains"
HRNET_OUT = OUT_DIR / "hrnet_w48_timm_features.pth"
RESNET_OUT = OUT_DIR / "resnet101_timm_features.pth"


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_packages():
    required = [
        ("torch", None),
        ("timm", "timm"),
        ("httpx", "httpx[socks]"),
        ("socksio", "socksio"),
        ("huggingface_hub", "huggingface_hub"),
        ("safetensors", "safetensors"),
    ]

    missing = []
    for module_name, package_name in required:
        try:
            importlib.import_module(module_name)
        except ImportError:
            if package_name is None:
                raise RuntimeError(
                    f"Required package '{module_name}' is missing. "
                    "Install PyTorch in the current conda environment first."
                )
            missing.append(package_name)

    if missing:
        print(f"[Setup] installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *missing])
        importlib.invalidate_caches()


def _is_loadable(path):
    if (not path.exists()) or (not path.is_file()):
        return False
    try:
        torch = importlib.import_module("torch")
        state = torch.load(path, map_location="cpu")
        return isinstance(state, dict) and len(state) > 0
    except Exception as exc:
        print(f"[Warn] existing file is not loadable, will redownload: {path} | {type(exc).__name__}: {exc}")
        return False


def save_timm_features(model_name, out_indices, out_path):
    torch = importlib.import_module("torch")
    timm = importlib.import_module("timm")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    force = _truthy(os.getenv("GCFNET_FORCE_PRETRAIN_DOWNLOAD", "0"))
    if (not force) and _is_loadable(out_path):
        print(f"[Skip] already exists: {out_path}")
        return

    print(f"[Download] timm model={model_name}")
    model = timm.create_model(
        model_name,
        pretrained=True,
        features_only=True,
        out_indices=tuple(out_indices),
    )
    torch.save(model.state_dict(), out_path)
    print(f"[Saved] {out_path}")


def main():
    print(f"[Pretrain] project root: {ROOT}")
    print(f"[Pretrain] output dir  : {OUT_DIR}")
    ensure_packages()

    save_timm_features(
        model_name="hrnet_w48",
        out_indices=(0, 1, 2, 3),
        out_path=HRNET_OUT,
    )
    save_timm_features(
        model_name="resnet101",
        out_indices=(1, 2, 3, 4),
        out_path=RESNET_OUT,
    )

    print("")
    print("[Done] local pretrained weights are ready.")
    print(f"HRNet : {HRNET_OUT}")
    print(f"ResNet: {RESNET_OUT}")


if __name__ == "__main__":
    main()
