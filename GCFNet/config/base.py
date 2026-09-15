import os
import ast


class BaseConfig:
    def __init__(self):
        # =========================
        # runtime
        # =========================
        self.device = "cuda"
        self.amp = True
        self.save_dir = "./checkpoints"
        self.save_name = "best_gcfnet_final.pth"
        self.save_last_checkpoint = True
        self.save_compact_checkpoint = False
        self.save_optimizer_state = True
        self.save_model_raw_state = True
        self.save_model_ema_state = True
        self.save_ema_shadow_state = True
        self.allow_checkpoint_save_error = False

        # =========================
        # training
        # =========================
        self.epochs = 30
        self.batch_size = 4
        self.lr = 2e-5
        self.weight_decay = 1e-4
        self.num_workers = 4
        self.max_grad_norm = 1.0
        self.seed = 42
        self.warmup_epochs = 3
        self.min_lr = 1e-6
        self.early_stop_patience = 8
        self.early_stop_min_delta = 1e-4
        self.early_stop_start_epoch = 0
        self.stage2_early_stop_grace_epochs = 2
        self.deterministic = False
        self.auto_oom_recovery = True
        self.oom_micro_batch_size = 1

        # EMA
        self.use_ema = True
        self.ema_decay = 0.999
        self.save_use_ema_weights = True
        self.resume_prefer_ema = True
        self.test_prefer_ema = True
        self.test_checkpoint_policy = "best"

        # dataloader
        self.persistent_workers = True
        self.prefetch_factor = 2

        # =========================
        # dataset split
        # =========================
        self.val_ratio = 0.2
        self.use_official_split_txt = False
        self.train_split_file = "train.txt"
        self.val_split_file = "val.txt"
        self.strict_txt_split_disjoint = True

        # =========================
        # loss
        # =========================
        self.ce_weight = 1.0
        self.dice_weight = 1.0
        self.focal_weight = 0.5
        self.focal_gamma = 2.0
        self.label_smoothing = 0.0
        self.ignore_index = 255
        self.include_ignore_label_as_class = False
        self.ignore_label_class_id = None

        # class weight
        self.compute_class_weights = True
        self.class_weights = None
        self.class_weight_min = 0.3
        self.class_weight_max = 6.0
        self.class_weight_boost_map = {}

        # deep supervision
        self.deep_supervision_aux_weight = 0.3

        # =========================
        # data augmentation
        # =========================
        self.normalize_opt = True
        self.normalize_sar = True
        self.train_normalize_stage = "pre"
        self.val_normalize_stage = "pre"
        self.train_sar_log_transform = False
        self.val_sar_log_transform = False

        self.enable_crop = True
        self.crop_size = 256
        self.use_class_aware_crop = True
        self.class_aware_crop_prob = 0.6
        self.class_aware_min_pixels = 48
        self.class_aware_max_try = 20

        self.flip_prob = 0.5
        self.rotate_prob = 0.5

        self.optical_jitter_prob = 0.5
        self.sar_noise_prob = 0.5

        # optical augmentation
        self.optical_brightness = 0.2
        self.optical_contrast = 0.2
        self.optical_noise_std = 0.01

        # SAR augmentation
        self.sar_noise_std = 0.03
        self.sar_speckle_std = 0.05

        # =========================
        # inference
        # =========================
        self.use_sliding_window = True
        self.use_tta = True
        self.eval_tta_start_epoch = 0
        self.ms_scales = (0.75, 1.0, 1.25)
        self.eval_ms_scales = self.ms_scales
        self.window_size = 256
        self.window_stride = 192

        # =========================
        # model
        # =========================
        self.deep_supervision = True
        self.decoder_dim = 192
        self.eca_kernel_size = 3
        self.gam_reduction = 4
        self.use_opt_modality = True
        self.use_sar_modality = True
        self.use_gmf_module = True
        self.use_gam_module = True
        self.gam_use_residual = True
        self.gmf_use_residual_sum = True
        self.pretrained_backbone = True
        self.opt_pretrained_path = "./pretrains/hrnet_w48_timm_features.pth"
        self.sar_pretrained_path = "./pretrains/resnet101_timm_features.pth"
        self.use_glvrm = True
        self.glvrm_decoder_stages = 1
        self.use_aspp_context = True
        self.aspp_dilations = (1, 3, 6)
        self.aspp_dropout = 0.05
        self.freeze_backbone_epochs = 0

        # =========================
        # dataset default fields
        # =========================
        self.dataset_name = ""
        self.root = ""
        self.num_classes = 8
        self.original_num_classes = 8
        self.enable_class_drop = False
        self.drop_classes = tuple()
        self.compact_class_ids = True
        self.class_id_map = {i: i for i in range(self.num_classes)}
        self.keep_classes = tuple(range(self.num_classes))

        self.optical_dir = "optical"
        self.sar_dir = "sar"
        self.mask_dir = "label"

        # Pohang settings
        self.train_scene = ""
        self.val_scene = ""

        # finetune / resume
        self.resume = True
        self.resume_path = "./checkpoints/best_gcfnet_final.pth"
        self.finetune_lr = 8e-6
        self.backbone_lr_mult = 0.2
        self.force_class_drop_scratch = True
        self.strict_resume_class_setup = True
        self.allow_resume_without_class_meta = False

        # ===== weighted sampler =====
        self.use_weighted_sampler = True
        self.rare_class_id = 7
        self.rare_class_boost = 4.0
        self.rare_target_ratio = 0.20
        # lower_bound: w=max(boost, required) (legacy)
        # match_target: w=required
        # upper_bound: w=min(boost, required)
        self.rare_target_ratio_mode = "lower_bound"
        self.rare_weight_power = 1.0
        self.min_rare_pixels = 20
        self.rare_sample_ref_pixels = 320
        self.rare_sample_max_weight = 4.0

        # =========================
        # copy-paste augmentation
        # =========================
        self.use_copypaste = True
        self.copypaste_prob = 0.5
        self.copypaste_target_class = 7
        self.copypaste_min_pixels = 10
        self.copypaste_max_paste = 1
        self.copypaste_patch_size = 128
        self.copypaste_cache_size = 192
        self.copypaste_source_sampling_power = 0.5

        # =========================
        # lovasz loss
        # =========================
        self.lovasz_weight = 0.15

        # rare-class Tversky loss
        self.rare_tversky_weight = 0.15
        self.rare_tversky_class = 7
        self.rare_tversky_alpha = 0.3
        self.rare_tversky_beta = 0.7

        # rare class margin loss
        self.rare_margin_weight = 0.05
        self.rare_margin_class = 7
        self.rare_margin = 0.5
        self.rare_margin_logit_clip = 20.0

        # confusion-targeted margin loss (focus on 7 -> 1/3 confusion)
        self.rare_confusion_weight = 0.0
        self.rare_confusion_target_class = 7
        self.rare_confusion_negative_classes = (1, 3)
        self.rare_confusion_margin = 0.35
        self.rare_confusion_topk = 2
        self.rare_confusion_max_violation = 6.0

        # stage-2 curriculum (late training: strengthen rare-class supervision)
        self.enable_stage2_curriculum = True
        self.stage2_start_ratio = 0.55
        self.stage2_lovasz_weight = 0.20
        self.stage2_rare_tversky_weight = 0.22
        self.stage2_rare_margin_weight = 0.08
        self.stage2_rare_confusion_weight = 0.05
        self.stage2_rare_confusion_negative_classes = (1, 2, 4, 6)
        self.stage2_deep_supervision_aux_weight = 0.25
        self.stage2_class_aware_crop_prob = 0.70
        self.stage2_copypaste_prob = 0.60
        self.stage2_copypaste_max_paste = 2
        self.stage2_rare_class_boost = 5.0
        self.stage2_rare_target_ratio = 0.26
        self.stage2_rare_target_ratio_mode = "lower_bound"
        self.stage2_rare_weight_power = 0.8
        self.stage2_rare_sample_max_weight = 6.0

        # label sanitization / diagnostics
        self.sanitize_labels = True
        self.report_label_quality = True
        self.label_quality_max_samples = 0
        self.skip_loss_if_gt = 80.0

        # inference class-bias search
        self.search_multi_class_bias = False
        self.logit_bias_dict = {
            0: -0.25,
            3: -0.5,
            7: 2.5,
        }
        self.auto_search_bias_if_no_file = False
        self.auto_search_bias_classes = (0,)
        self.auto_search_bias_values = (0.0, 0.1, 0.2, 0.3)
        self.auto_search_bias_space = {}
        self.auto_search_bias_rounds = 1
        self.auto_search_bias_min_delta = 0.0

        # progress display
        self.use_tqdm = True
        self.show_val_progress = False
        self.progress_ascii = True
        self.progress_mininterval = 0.5
        self.progress_ncols = 120
        self.progress_disable_non_tty = True

        self.print_confusion = True
        self.confusion_class_ids = (0, 3, 7)
        self.metric_ignore_absent_classes = False

        # save searched bias result
        self.save_bias_search_result = True
        self.bias_result_save_path = "./checkpoints/best_bias.json"
        self.auto_load_bias_result = False
        self.bias_result_load_path = "./checkpoints/best_bias.json"
        self.strict_bias_class_setup = True
        self.allow_bias_without_meta = False
        self.eval_report_without_bias = True
        self.bias_effect_warn_threshold = 0.015

    def _remap_class_id(self, cls_id):
        try:
            cls_id = int(cls_id)
        except Exception:
            return None
        return self.class_id_map.get(cls_id, None)

    def apply_class_drop_policy(self):
        base_num_classes = int(getattr(self, "num_classes", 0))
        self.original_num_classes = base_num_classes

        raw_drop = getattr(self, "drop_classes", tuple())
        drop = []
        for c in raw_drop:
            try:
                ci = int(c)
            except Exception:
                continue
            if 0 <= ci < base_num_classes:
                drop.append(ci)
        drop = tuple(sorted(set(drop)))
        self.drop_classes = drop

        keep_classes = tuple(c for c in range(base_num_classes) if c not in drop)
        self.keep_classes = keep_classes

        if (not bool(getattr(self, "enable_class_drop", False))) or len(drop) == 0:
            self.class_id_map = {c: c for c in range(base_num_classes)}
            return

        if bool(getattr(self, "compact_class_ids", True)):
            self.class_id_map = {old: new for new, old in enumerate(keep_classes)}
            self.num_classes = len(keep_classes)
        else:
            self.class_id_map = {old: old for old in keep_classes}
            self.num_classes = base_num_classes

        remapped_rare = self._remap_class_id(getattr(self, "rare_class_id", None))
        if remapped_rare is None:
            self.use_weighted_sampler = False
            self.use_copypaste = False
            self.use_class_aware_crop = False
            self.enable_stage2_curriculum = False

            self.rare_tversky_weight = 0.0
            self.rare_margin_weight = 0.0
            self.rare_confusion_weight = 0.0
            self.rare_class_id = -1
            self.copypaste_target_class = -1
            self.rare_tversky_class = -1
            self.rare_margin_class = -1
            self.rare_confusion_target_class = -1
        else:
            self.rare_class_id = remapped_rare
            self.copypaste_target_class = remapped_rare
            self.rare_tversky_class = remapped_rare
            self.rare_margin_class = remapped_rare
            self.rare_confusion_target_class = remapped_rare

            remap_neg = []
            for c in getattr(self, "rare_confusion_negative_classes", tuple()):
                mc = self._remap_class_id(c)
                if mc is not None and mc != remapped_rare and mc not in remap_neg:
                    remap_neg.append(mc)
            self.rare_confusion_negative_classes = tuple(remap_neg)

            remap_stage2_neg = []
            for c in getattr(self, "stage2_rare_confusion_negative_classes", tuple()):
                mc = self._remap_class_id(c)
                if mc is not None and mc != remapped_rare and mc not in remap_stage2_neg:
                    remap_stage2_neg.append(mc)
            self.stage2_rare_confusion_negative_classes = tuple(remap_stage2_neg)

        remap_bias = {}
        for k, v in getattr(self, "logit_bias_dict", {}).items():
            mk = self._remap_class_id(k)
            if mk is not None:
                remap_bias[int(mk)] = float(v)
        self.logit_bias_dict = remap_bias

        remap_auto_search_classes = []
        for c in getattr(self, "auto_search_bias_classes", tuple()):
            mc = self._remap_class_id(c)
            if mc is not None and mc not in remap_auto_search_classes:
                remap_auto_search_classes.append(mc)
        self.auto_search_bias_classes = tuple(remap_auto_search_classes)

        raw_auto_search_space = getattr(self, "auto_search_bias_space", {})
        remap_auto_search_space = {}
        if isinstance(raw_auto_search_space, dict):
            for k, values in raw_auto_search_space.items():
                mk = self._remap_class_id(k)
                if mk is None:
                    continue
                if not isinstance(values, (list, tuple)):
                    values = [values]
                parsed = []
                for v in values:
                    parsed.append(float(v))
                if len(parsed) > 0:
                    remap_auto_search_space[int(mk)] = tuple(parsed)
        self.auto_search_bias_space = remap_auto_search_space

        remap_confusion = []
        for c in getattr(self, "confusion_class_ids", tuple()):
            mc = self._remap_class_id(c)
            if mc is not None and mc not in remap_confusion:
                remap_confusion.append(mc)
        self.confusion_class_ids = tuple(remap_confusion)

        # Classifier head shape usually changes after class dropping.
        # Keep previous safe behavior by default, but allow explicit override for continued finetuning.
        if bool(getattr(self, "force_class_drop_scratch", True)):
            self.resume = False
        self.early_stop_start_epoch = max(int(getattr(self, "early_stop_start_epoch", 0)), 6)

    def _parse_env_value(self, raw_value, current_value):
        if isinstance(current_value, bool):
            v = str(raw_value).strip().lower()
            if v in {"1", "true", "yes", "y", "on"}:
                return True
            if v in {"0", "false", "no", "n", "off"}:
                return False
            raise ValueError(f"Cannot parse bool from '{raw_value}'")

        if isinstance(current_value, int) and not isinstance(current_value, bool):
            return int(raw_value)

        if isinstance(current_value, float):
            return float(raw_value)

        if isinstance(current_value, str):
            return str(raw_value)

        if current_value is None:
            try:
                return ast.literal_eval(raw_value)
            except Exception:
                return raw_value

        if isinstance(current_value, tuple):
            parsed = ast.literal_eval(raw_value)
            return tuple(parsed)

        if isinstance(current_value, list):
            parsed = ast.literal_eval(raw_value)
            return list(parsed)

        if isinstance(current_value, dict):
            parsed = ast.literal_eval(raw_value)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected dict, got: {type(parsed)}")
            return parsed

        try:
            return ast.literal_eval(raw_value)
        except Exception:
            return raw_value

    def apply_env_overrides(self, prefix="GCFNET_"):
        attr_lut = {k.lower(): k for k in vars(self).keys()}
        applied = []

        for env_k, env_v in os.environ.items():
            if not env_k.startswith(prefix):
                continue

            key = env_k[len(prefix):].lower()
            if key not in attr_lut:
                continue

            attr_name = attr_lut[key]
            current = getattr(self, attr_name)
            new_value = self._parse_env_value(env_v, current)
            setattr(self, attr_name, new_value)
            applied.append((attr_name, current, new_value))

        if applied:
            print("[Config override] from environment:")
            for name, old_v, new_v in applied:
                print(f" - {name}: {old_v!r} -> {new_v!r}")

        return applied

    @property
    def save_path(self):
        return os.path.join(self.save_dir, self.save_name)

    def check(self):
        if not self.root:
            raise ValueError("Config.root is empty.")
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        os.makedirs(self.save_dir, exist_ok=True)
