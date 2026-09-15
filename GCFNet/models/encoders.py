import torch
import torch.nn as nn
import timm


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_ema", "model_state_dict"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
    return ckpt


def _strip_prefixes(state_dict, prefixes):
    out = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        out[new_key] = value
    return out


def _match_count(module, state_dict):
    target = module.state_dict()
    count = 0
    for key, value in state_dict.items():
        if key in target and tuple(target[key].shape) == tuple(value.shape):
            count += 1
    return count


def _load_local_pretrained(module, path, name):
    path = str(path or "").strip()
    if len(path) == 0:
        return False
    try:
        ckpt = torch.load(path, map_location="cpu")
        raw_state = _extract_state_dict(ckpt)
        if not isinstance(raw_state, dict):
            raise RuntimeError("checkpoint does not contain a state_dict")

        candidates = [
            raw_state,
            _strip_prefixes(raw_state, ("module.", "model.", "backbone.")),
            _strip_prefixes(raw_state, ("module.model.", "module.backbone.")),
        ]
        best_state = max(candidates, key=lambda sd: _match_count(module, sd))
        matched = _match_count(module, best_state)
        if matched <= 0:
            raise RuntimeError("no matching parameter keys")
        incompatible = module.load_state_dict(best_state, strict=False)
        print(
            f"[Pretrain] loaded local {name}: {path} | "
            f"matched={matched}, missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        return True
    except Exception as e:
        print(f"[WARN] local {name} pretrained load failed: {path} | {type(e).__name__}: {e}")
        return False


class HRNetW48Encoder(nn.Module):
    def __init__(self, pretrained: bool = False, pretrained_path: str = ""):
        super().__init__()
        use_timm_pretrained = bool(pretrained) and len(str(pretrained_path or "").strip()) == 0
        try:
            self.backbone = timm.create_model(
                "hrnet_w48",
                pretrained=use_timm_pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
        except Exception as e:
            print(f"[WARN] HRNet pretrained load failed: {e}")
            print("[WARN] Falling back to pretrained=False")
            self.backbone = timm.create_model(
                "hrnet_w48",
                pretrained=False,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )

        if len(str(pretrained_path or "").strip()) > 0:
            _load_local_pretrained(self.backbone, pretrained_path, "HRNet-W48")

        info = self.backbone.feature_info.get_dicts()
        self.out_channels = [d["num_chs"] for d in info]

    def forward(self, x):
        return self.backbone(x)


class ResNet101Encoder(nn.Module):
    def __init__(self, in_ch: int = 1, pretrained: bool = False, pretrained_path: str = ""):
        super().__init__()
        self.input_proj = nn.Identity() if in_ch == 3 else nn.Conv2d(in_ch, 3, kernel_size=1)

        use_timm_pretrained = bool(pretrained) and len(str(pretrained_path or "").strip()) == 0
        try:
            self.backbone = timm.create_model(
                "resnet101",
                pretrained=use_timm_pretrained,
                features_only=True,
                out_indices=(1, 2, 3, 4),
            )
        except Exception as e:
            print(f"[WARN] ResNet pretrained load failed: {e}")
            print("[WARN] Falling back to pretrained=False")
            self.backbone = timm.create_model(
                "resnet101",
                pretrained=False,
                features_only=True,
                out_indices=(1, 2, 3, 4),
            )

        if len(str(pretrained_path or "").strip()) > 0:
            _load_local_pretrained(self.backbone, pretrained_path, "ResNet101")

        info = self.backbone.feature_info.get_dicts()
        self.out_channels = [d["num_chs"] for d in info]

    def forward(self, x):
        x = self.input_proj(x)
        return self.backbone(x)
