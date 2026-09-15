import os

from .base import BaseConfig


class YESegConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.dataset_name = "YESeg-OPT-SAR"
        self.root = os.getenv("YESEG_ROOT", "/home/featurize/data/YESeg_OPT_SAR").strip()
        self.num_classes = 8
        self.original_num_classes = 8

        # One-shot 256x256 profile targeting about 78 mIoU from scratch.
        # Keep crop/window at 256; the gains come from steadier optimization, not 512 inputs.
        self.epochs = 150
        self.batch_size = 4
        self.lr = 5.5e-5
        self.min_lr = 3.0e-7
        self.weight_decay = 1.2e-4
        self.warmup_epochs = 8
        self.early_stop_patience = 36
        self.early_stop_start_epoch = 40
        self.ema_decay = 0.9996
        self.skip_loss_if_gt = 30.0
        self.resume = False
        self.force_class_drop_scratch = False
        self.resume_path = "./checkpoints/best_gcfnet_final.pth"
        self.finetune_lr = float(self.lr)
        self.backbone_lr_mult = 0.20
        self.freeze_backbone_epochs = 3
        self.test_checkpoint_policy = "best"
        self.pretrained_backbone = False
        self.opt_pretrained_path = "./pretrains/hrnet_w48_timm_features.pth"
        self.sar_pretrained_path = "./pretrains/resnet101_timm_features.pth"
        self.use_aspp_context = True
        self.aspp_dropout = 0.05
        self.use_glvrm = True
        self.glvrm_decoder_stages = 1
        # Note: use original class ids before class-drop remap.
        self.logit_bias_dict = {1: 0.1, 2: 0.1}
        self.auto_load_bias_result = True
        self.auto_search_bias_if_no_file = True
        self.auto_search_bias_classes = (1, 2, 3)
        self.auto_search_bias_values = (-0.2, -0.1, 0.0, 0.1, 0.2)
        self.auto_search_bias_space = {
            1: (-0.3, -0.2, -0.1, 0.0, 0.1),
            2: (-0.2, -0.1, 0.0, 0.1, 0.2),
            3: (-0.2, -0.1, 0.0, 0.1),
        }
        self.auto_search_bias_rounds = 2
        self.auto_search_bias_min_delta = 1e-4
        self.confusion_class_ids = (1, 2, 3)
        self.bias_result_save_path = "./checkpoints/best_bias_drop0_7.json"
        self.bias_result_load_path = "./checkpoints/best_bias_drop0_7.json"

        # Loss setup: keep weighted CE + Dice + mild focal for hard pixels.
        self.compute_class_weights = True
        self.class_weight_boost_map = {2: 1.08, 4: 1.08}
        self.label_smoothing = 0.015
        self.ce_weight = 1.0
        self.dice_weight = 1.0
        self.focal_weight = 0.10
        self.lovasz_weight = 0.10
        self.rare_tversky_weight = 0.0
        self.rare_margin_weight = 0.0
        self.rare_confusion_weight = 0.0

        # Flip-TTA stabilizes 256x256 validation without changing the train split.
        # Training-time validation stays single-scale; final test uses the stronger ms_scales below.
        self.use_sliding_window = True
        self.use_tta = True
        self.eval_tta_start_epoch = 60
        self.eval_ms_scales = (1.0,)
        self.ms_scales = (0.75, 1.0, 1.25)
        self.window_size = 256
        self.window_stride = 192

        # Train augmentations: keep all crops at 256 while increasing tail-class exposure.
        self.enable_crop = True
        self.crop_size = 256
        self.flip_prob = 0.5
        self.rotate_prob = 0.5
        self.optical_jitter_prob = 0.45
        self.sar_noise_prob = 0.40

        # Keep modality normalization aligned with pretrained/previous best behavior.
        self.normalize_opt = True
        self.normalize_sar = True
        self.train_normalize_stage = "pre"
        self.val_normalize_stage = "pre"
        self.train_sar_log_transform = False
        self.val_sar_log_transform = False

        self.persistent_workers = True
        self.prefetch_factor = 2

        # Class-drop setup for the current 6-class target.
        self.enable_class_drop = True
        self.drop_classes = (0, 7)
        self.compact_class_ids = True

        # Original class 5 maps to train class 4 after dropping (0, 7).
        self.use_weighted_sampler = True
        self.rare_class_id = 5
        self.rare_class_boost = 2.6
        self.rare_target_ratio = 0.15
        self.rare_target_ratio_mode = "upper_bound"
        self.rare_weight_power = 0.75
        self.min_rare_pixels = 24
        self.rare_sample_ref_pixels = 192
        self.rare_sample_max_weight = 3.8
        self.use_class_aware_crop = True
        self.class_aware_crop_prob = 0.25
        self.class_aware_min_pixels = 24
        self.class_aware_max_try = 18
        self.use_copypaste = False

        # Late stage is intentionally mild: improve tail classes without destabilizing the head classes.
        self.enable_stage2_curriculum = True
        self.stage2_start_ratio = 0.52
        self.stage2_lovasz_weight = 0.12
        self.stage2_rare_tversky_weight = 0.06
        self.stage2_rare_margin_weight = 0.015
        self.stage2_rare_confusion_weight = 0.0
        self.stage2_deep_supervision_aux_weight = 0.18
        self.stage2_class_aware_crop_prob = 0.45
        self.stage2_rare_class_boost = 2.8
        self.stage2_rare_target_ratio = 0.17
        self.stage2_rare_target_ratio_mode = "upper_bound"
        self.stage2_rare_weight_power = 0.75
        self.stage2_rare_sample_max_weight = 4.0

        self.optical_dir = "optical"
        self.sar_dir = "sar"
        self.mask_dir = "label"

        # Force official split txt usage.
        self.use_official_split_txt = True
        self.train_split_file = "train.txt"
        self.val_split_file = "val.txt"

        self.apply_env_overrides()
        self.apply_class_drop_policy()
