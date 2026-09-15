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
- Default checkpoint name: `checkpoints/best_gcfnet_pohang_to90.pth`

## Quick Start

### 1. Prepare environment

Recommended software stack:

- Python 3.10+
- PyTorch
- torchvision
- timm
- numpy
- Pillow
- tqdm
- rasterio
- huggingface-hub and safetensors, if downloading pretrained backbones with `tools/download_pretrained_weights.py`

Install additional packages as needed if your local environment differs.

### 2. Set dataset roots

```bash
export YESEG_ROOT=/home/featurize/data/YESeg_OPT_SAR
export POHANG_ROOT=/home/featurize/data/Pohang
```

These are also the built-in defaults in `config/yeseg.py` and `config/pohang.py`, so the exports are optional on the current Featurize server.

### 3. Train

```bash
python tools/train.py --dataset yeseg
python tools/train.py --dataset pohang
```

For Pohang, generate the split files before training:

```bash
python tools/make_pohang_split.py --root /path/to/Pohang --mode block --scenes cloud,no_cloud --val-ratio 0.2 --seed 42 --block-size 32 --gap-blocks 1
```

On the current server:

```bash
python tools/make_pohang_split.py --root /home/featurize/data/Pohang --mode block --scenes cloud,no_cloud --val-ratio 0.2 --seed 42 --block-size 32 --gap-blocks 1
```

Use `--mode block` if you need both `cloud` and `no_cloud` scenes in training while still keeping contiguous tile blocks out of the training set around validation blocks.

### 4. Evaluate

```bash
python tools/test.py --dataset yeseg
python tools/test.py --dataset pohang
```

### 5. Profile model complexity

```bash
python tools/profile_model.py --dataset yeseg --device cuda
```

## Configuration Philosophy

The base configuration in `config/base.py` is intentionally verbose. It exposes:

- optimization hyperparameters
- class-remapping and class-drop policy
- class-weight computation
- augmentation settings
- rare-class sampling and loss controls
- curriculum-stage overrides
- inference-time bias search
- progress and diagnostics controls

Any field can be overridden through `GCFNET_*` environment variables. For example:

```bash
export GCFNET_EPOCHS=40
export GCFNET_BATCH_SIZE=2
export GCFNET_RESUME=false
python tools/train.py --dataset pohang
```

This makes the artifact practical for ablation studies without duplicating configuration files.

## Reproducibility Notes

- The training and evaluation scripts record class-setup metadata inside checkpoints to guard against incompatible resume/evaluation.
- The test script checks split integrity for txt-based protocols before reporting metrics.
- Bytecode caches, IDE settings, and intermediate artifacts are intentionally excluded from the source release.

## Reporting Guidance

For paper writing or supplementary materials, we recommend reporting:

- dataset profile (`yeseg` or `pohang`)
- raw-to-train class mapping
- checkpoint selection policy (`best`, `latest`, or `best_or_latest`)
- whether EMA weights are used for evaluation
- augmentation and sampler settings
- whether bias-search results were loaded at test time

These choices materially affect comparability across runs.

## Citation

Please add the final paper citation before public release.
