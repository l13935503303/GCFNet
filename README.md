# GCFNet

GCFNet (Gated Cross-modal Fusion Network) is a dual-modal semantic segmentation framework for optical-SAR remote sensing scenes. This release is organized as a paper-oriented research artifact: it keeps the core model implementation, dataset-specific configurations, and training/evaluation scripts for reproducibility-focused use.

## Highlights

- Dual-stream encoder design with an optical HRNet-W48 branch and a SAR ResNet-101 branch.
- Cross-modal feature interaction through gated multi-modal fusion (`GMF`) and global attention modulation (`GAMBlock`).
- A decoder pipeline with `GLVRMBlock`, skip-connected decoding stages, and optional deep supervision.
- Two built-in experiment profiles:
  - `YESeg-OPT-SAR`: 8 raw classes compacted into 6 training classes by dropping class `0` and `7`.
  - `Pohang`: 6-class setup in which raw label `255` is a valid semantic label and is mapped to class ID `5` (`Other`).
- Reproducibility-oriented training logic, including checkpoint metadata, class-remapping safety checks, optional EMA evaluation, and dataset-specific evaluation policies.

## Repository Layout

- `config/`: dataset-aware training and evaluation configurations.
- `dataset/`: sample construction, label remapping, augmentations, and dataset wrappers.
- `models/`: GCFNet model definition and component modules.
- `tools/train.py`: main training entry for `yeseg` and `pohang`.
- `tools/test.py`: evaluation entry with class-setup and split integrity checks.
- `tools/profile_model.py`: parameter, FLOPs, and latency profiling helper.

## Method Overview

The implementation in `models/gcfnet.py` exposes `models.GCFNet` and follows a staged dual-modal fusion design:

1. The optical stream extracts multi-scale features using `HRNetW48Encoder`.
2. The SAR stream extracts modality-specific features using `ResNet101Encoder`.
3. Channel alignment and spatial alignment are applied before fusion.
4. `GMF` fuses modality pairs at each scale and `GAMBlock` refines the fused representation.
5. A neck module based on `GLVRMBlock` prepares the top-level representation for decoding.
6. `DecoderStage` modules progressively recover spatial detail with skip connections from fused encoder features.
7. The final segmentation head predicts logits at the input resolution, with optional auxiliary outputs for deep supervision.

This layout maps cleanly to an academic method section: modality-specific encoders, gated cross-modal fusion, attention refinement, and hierarchical decoding are separated into explicit modules.

## Supported Experiment Profiles

### YESeg-OPT-SAR

- Config file: `config/yeseg.py`
- Environment variable: `YESEG_ROOT`
- Split protocol: official `train.txt` / `val.txt`
- Training classes: 6 compact classes after dropping raw classes `0` and `7`
- Default checkpoint name: `checkpoints/best_gcfnet_final.pth`
- Optional bias search result: `checkpoints/best_bias_drop0_7.json`

### Pohang

- Config file: `config/pohang.py`
- Environment variable: `POHANG_ROOT`
- Training classes: ID `0` = Buildings, `1` = Farmland, `2` = Greenery, `3` = Water, `4` = Roads, and `5` = Other; raw label `255` is remapped to class ID `5` (`Other`) rather than ignored.
- Split protocol: `train_pohang.txt` / `val_pohang.txt` are generated in `block` mode from both `cloud` and `no_cloud` scenes with validation ratio `0.2`, seed `42`, block size `32`, and one adjacent gap block excluded from training around each validation block.
- Default checkpoint name: `checkpoints/best_gcfnet_pohang.pth`


