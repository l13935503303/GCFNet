from . import dataset as _d


class _IdentityTransform:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, sample):
        return sample


build_yeseg_samples = _d.build_yeseg_samples
build_pohang_samples = getattr(_d, "build_pohang_samples", None)
split_samples = _d.split_samples
split_samples_by_txt = getattr(_d, "split_samples_by_txt", None)

MultimodalSegDataset = _d.MultimodalSegDataset
Compose = _d.Compose
RandomFlipRotate = _d.RandomFlipRotate

RandomCropPair = getattr(_d, "RandomCropPair", _IdentityTransform)
ClassAwareRandomCrop = getattr(_d, "ClassAwareRandomCrop", _IdentityTransform)
ProbabilisticClassAwareCrop = getattr(_d, "ProbabilisticClassAwareCrop", _IdentityTransform)
TargetClassCopyPaste = getattr(_d, "TargetClassCopyPaste", _IdentityTransform)
sanitize_mask_labels = getattr(_d, "sanitize_mask_labels", None)
remap_mask_labels = getattr(_d, "remap_mask_labels", None)

OpticalColorJitter = getattr(_d, "OpticalColorJitter", _IdentityTransform)
SARNoiseAug = getattr(_d, "SARNoiseAug", _IdentityTransform)
SARNoise = getattr(_d, "SARNoise", SARNoiseAug)

__all__ = [
    "build_yeseg_samples",
    "build_pohang_samples",
    "split_samples",
    "split_samples_by_txt",
    "MultimodalSegDataset",
    "Compose",
    "RandomFlipRotate",
    "RandomCropPair",
    "ClassAwareRandomCrop",
    "ProbabilisticClassAwareCrop",
    "TargetClassCopyPaste",
    "sanitize_mask_labels",
    "remap_mask_labels",
    "OpticalColorJitter",
    "SARNoiseAug",
    "SARNoise",
]
