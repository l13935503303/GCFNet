import os

try:
    from .base import BaseConfig
except ImportError:  # Allow running this file directly: python config/pohang.py
    from base import BaseConfig


class PohangConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.dataset_name = "Pohang"

        env_root = os.getenv("POHANG_ROOT", "/home/featurize/data/Pohang").strip()
        self.root = env_root
        self.num_classes = 6
        # Raw label 255 is a valid semantic label and is remapped to class id=5 (Other).
        self.include_ignore_label_as_class = True
        self.ignore_label_class_id = 5

        self.train_scene = "cloud"
        self.val_scene = "no_cloud"
        self.pohang_split_scenes = ("cloud", "no_cloud")

        # Pohang-to90 scratch recipe on the leakage-aware block split.
        # This is intentionally stronger than the quick profile: enough time for
        # the hard class-1/class-5 boundaries to stabilize without using resume.
        self.epochs = 120
        self.warmup_epochs = 5
        self.early_stop_start_epoch = 45
        self.early_stop_patience = 28
        self.lr = 4.0e-5
        self.min_lr = 5e-7
        self.weight_decay = 1.5e-4
        self.ema_decay = 0.9995
        self.freeze_backbone_epochs = 1

        # Loss balance: focus the weak IoU classes from the 88.19 run
        # (class 1 and class 5) without over-sampling frequent class-5 tiles.
        self.compute_class_weights = True
        self.class_weight_min = 0.35
        self.class_weight_max = 2.4
        self.class_weight_boost_map = {1: 1.06, 5: 1.16}
        self.label_smoothing = 0.015
        self.ce_weight = 1.0
        self.dice_weight = 1.0
        self.focal_weight = 0.08
        self.lovasz_weight = 0.12
        self.deep_supervision_aux_weight = 0.18

        # Class-specific auxiliary losses. Class 5 is not rare in this split,
        # so keep these mild; the goal is boundary polishing, not distribution shift.
        self.rare_tversky_weight = 0.04
        self.rare_tversky_class = 5
        self.rare_tversky_alpha = 0.3
        self.rare_tversky_beta = 0.7
        self.rare_margin_weight = 0.01
        self.rare_margin_class = 5
        self.rare_margin = 0.30
        self.rare_margin_logit_clip = 20.0
        self.rare_confusion_weight = 0.006
        self.rare_confusion_target_class = 5
        self.rare_confusion_negative_classes = (0, 1, 2, 4)
        self.rare_confusion_margin = 0.30
        self.rare_confusion_topk = 2
        self.rare_confusion_max_violation = 6.0

        # Augment profile.
        self.enable_crop = True
        self.crop_size = 256
        self.flip_prob = 0.5
        self.rotate_prob = 0.4
        self.optical_jitter_prob = 0.25
        self.sar_noise_prob = 0.20
        self.normalize_opt = True
        self.normalize_sar = True
        self.train_normalize_stage = "pre"
        self.val_normalize_stage = "pre"
        self.train_sar_log_transform = False
        self.val_sar_log_transform = False

        # Pohang should use an explicit split file. Patch-level random splitting
        # can place neighbouring tiles from the same source area in both train
        # and validation sets, which makes mIoU look overly optimistic.
        self.use_official_split_txt = True
        self.val_ratio = 0.2
        self.train_split_file = "train_pohang.txt"
        self.val_split_file = "val_pohang.txt"
        self.strict_txt_split_disjoint = True

        # Save names isolated from the existing 88.19 checkpoint.
        self.save_name = "best_gcfnet_pohang_to90.pth"
        self.last_save_name = "last_gcfnet_pohang_to90.pth"

        # Train from scratch by default for Pohang.
        # If best checkpoint exists and you want stage-2 finetune:
        # set GCFNET_RESUME=true and GCFNET_FINETUNE_LR=3e-6 in shell.
        self.resume = False
        self.resume_path = "./checkpoints/best_gcfnet_pohang.pth"
        self.finetune_lr = float(self.lr)
        self.force_class_drop_scratch = False
        self.enable_class_drop = False
        self.drop_classes = tuple()

        # Avoid class-5 weighted sampling: in the block split it appears in most
        # training tiles, so treating it as "rare" lowers useful samples.
        self.use_weighted_sampler = False
        self.rare_class_id = 5
        self.rare_class_boost = 1.0
        self.rare_target_ratio = 0.0
        self.rare_target_ratio_mode = "upper_bound"
        self.rare_weight_power = 1.0
        self.min_rare_pixels = 16
        self.rare_sample_ref_pixels = 192
        self.rare_sample_max_weight = 1.0

        self.use_class_aware_crop = False
        self.class_aware_crop_prob = 0.0
        self.class_aware_min_pixels = 16
        self.class_aware_max_try = 18

        self.use_copypaste = False
        self.copypaste_target_class = -1

        # Late curriculum: add mild class-5 polishing after representation stabilizes.
        self.enable_stage2_curriculum = True
        self.stage2_start_ratio = 0.55
        self.stage2_lovasz_weight = 0.14
        self.stage2_rare_tversky_weight = 0.06
        self.stage2_rare_margin_weight = 0.015
        self.stage2_rare_confusion_weight = 0.01
        self.stage2_rare_confusion_negative_classes = (0, 1, 2, 4)
        self.stage2_deep_supervision_aux_weight = 0.14
        self.stage2_class_aware_crop_prob = 0.0
        self.stage2_rare_class_boost = 1.0
        self.stage2_rare_target_ratio = 0.0
        self.stage2_rare_target_ratio_mode = "upper_bound"
        self.stage2_rare_weight_power = 1.0
        self.stage2_rare_sample_max_weight = 1.0

        # Disable bias file auto-logic for Pohang by default.
        self.auto_load_bias_result = False
        self.auto_search_bias_if_no_file = False
        self.logit_bias_dict = {}
        self.bias_result_save_path = "./checkpoints/best_bias_pohang.json"
        self.bias_result_load_path = "./checkpoints/best_bias_pohang.json"

        # Strong evaluation profile for the 256x256 Pohang split.
        self.show_val_progress = True
        self.use_sliding_window = True
        self.use_tta = True
        # Training validation uses only scale 1.0 for speed; test.py keeps ms_scales below.
        self.eval_ms_scales = (1.0,)
        self.eval_tta_start_epoch = 60
        self.ms_scales = (0.75, 1.0, 1.25)
        self.window_size = 256
        self.window_stride = 256
        self.confusion_class_ids = (0, 1, 2, 3, 4, 5)
        self.metric_ignore_absent_classes = False

        self.apply_env_overrides()
        self.apply_class_drop_policy()
