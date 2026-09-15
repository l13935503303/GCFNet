import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Loss
# =========================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, class_weights=None, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

        if class_weights is not None:
            self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        valid_mask = (target >= 0) & (target != self.ignore_index)
        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        target_1h = F.one_hot(
            target_safe.long(),
            num_classes,
        ).permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask
        target_1h = target_1h * valid_mask

        dims = (0, 2, 3)

        inter = (probs * target_1h).sum(dims)
        denom = probs.sum(dims) + target_1h.sum(dims)

        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)

        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            dice_loss = dice_loss * w
            return dice_loss.sum() / w.sum().clamp_min(1e-8)

        return dice_loss.mean()

class RareClassTverskyLoss(nn.Module):
    def __init__(
        self,
        target_class=7,
        alpha=0.3,
        beta=0.7,
        smooth=1.0,
        ignore_index=255,
    ):
        super().__init__()
        self.target_class = target_class
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        logits: [B, C, H, W]
        target: [B, H, W]
        """
        probs = torch.softmax(logits, dim=1)
        prob_cls = probs[:, self.target_class, :, :]

        valid = target != self.ignore_index
        gt_cls = (target == self.target_class).float()

        prob_cls = prob_cls * valid.float()
        gt_cls = gt_cls * valid.float()

        tp = (prob_cls * gt_cls).sum()
        fp = (prob_cls * (1.0 - gt_cls)).sum()
        fn = ((1.0 - prob_cls) * gt_cls).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return 1.0 - tversky

class RareClassMarginLoss(nn.Module):
    def __init__(
        self,
        target_class=7,
        margin=0.5,
        logit_clip=20.0,
        ignore_index=255,
    ):
        super().__init__()
        self.target_class = target_class
        self.margin = margin
        self.logit_clip = float(logit_clip)
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        鍙 target_class 鐨勭湡瀹炲儚绱犺绠?margin CE銆?        logits: [B, C, H, W]
        target: [B, H, W]
        """
        valid = target != self.ignore_index
        rare_mask = valid & (target == self.target_class)

        if rare_mask.sum() == 0:
            return logits.sum() * 0.0

        adjusted_logits = logits.clone()

        # 瀵圭8绫?logit 鍑?margin锛岃妯″瀷蹇呴』瀛﹀嚭鏇撮珮缃俊搴?        adjusted_logits[:, self.target_class, :, :] -= self.margin
        if self.logit_clip > 0:
            adjusted_logits = adjusted_logits.clamp(min=-self.logit_clip, max=self.logit_clip)

        # 鍙彇绗?绫诲儚绱犱綅缃绠?CE
        logits_flat = adjusted_logits.permute(0, 2, 3, 1).contiguous()
        logits_rare = logits_flat[rare_mask]
        target_rare = target[rare_mask]

        return F.cross_entropy(logits_rare, target_rare)


class RareConfusionMarginLoss(nn.Module):
    def __init__(
        self,
        target_class=7,
        negative_classes=(1, 3),
        margin=0.35,
        topk=2,
        max_violation=6.0,
        ignore_index=255,
    ):
        super().__init__()
        self.target_class = int(target_class)
        self.negative_classes = tuple(int(c) for c in negative_classes if int(c) != int(target_class))
        self.margin = float(margin)
        self.topk = max(int(topk), 1)
        self.max_violation = float(max_violation)
        self.ignore_index = int(ignore_index)

    def forward(self, logits, target):
        """
        Encourage GT target_class logits to be larger than confusing classes by margin.
        """
        valid = target != self.ignore_index
        target_mask = valid & (target == self.target_class)
        if target_mask.sum() == 0 or len(self.negative_classes) == 0:
            return logits.sum() * 0.0

        logits_flat = logits.permute(0, 2, 3, 1).contiguous()
        sel_logits = logits_flat[target_mask]  # [N, C]

        pos = sel_logits[:, self.target_class]
        neg_logits = sel_logits[:, list(self.negative_classes)]  # [N, K]
        k = min(self.topk, neg_logits.shape[1])
        hard_neg, _ = torch.topk(neg_logits, k=k, dim=1, largest=True, sorted=False)

        margins = self.margin - (pos.unsqueeze(1) - hard_neg)
        margins = F.relu(margins)
        if self.max_violation > 0:
            margins = torch.clamp(margins, max=self.max_violation)
        return margins.mean()

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors.
    gt_sorted: [P]
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()

    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-8)

    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

    return jaccard


def flatten_probas(probas, labels, ignore_index=255):
    """
    probas: [B, C, H, W]
    labels: [B, H, W]
    return:
        probas_flat: [P, C]
        labels_flat: [P]
    """
    if probas.dim() == 4:
        probas = probas.permute(0, 2, 3, 1).contiguous()

    c = probas.size(-1)
    probas = probas.view(-1, c)
    labels = labels.view(-1)

    valid = labels != ignore_index
    probas = probas[valid]
    labels = labels[valid]

    return probas, labels


def lovasz_softmax_flat(probas, labels, classes="present"):
    """
    Multi-class Lovasz-Softmax loss.
    probas: [P, C]
    labels: [P]
    """
    if probas.numel() == 0:
        return probas.sum() * 0.0

    c = probas.size(1)
    losses = []

    class_to_sum = list(range(c)) if classes in ["all", "present"] else classes

    for cls in class_to_sum:
        fg = (labels == cls).float()

        if classes == "present" and fg.sum() == 0:
            continue

        class_pred = probas[:, cls]
        errors = (fg - class_pred).abs()

        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]

        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    if len(losses) == 0:
        return probas.sum() * 0.0

    return torch.stack(losses).mean()


def lovasz_softmax(logits, labels, ignore_index=255, classes="present"):
    """
    logits: [B, C, H, W]
    labels: [B, H, W]
    """
    probas = F.softmax(logits, dim=1)
    probas, labels = flatten_probas(probas, labels, ignore_index)
    return lovasz_softmax_flat(probas, labels, classes=classes)

class SegLoss(nn.Module):
    def __init__(
        self,
        ignore_index=255,
        ce_weight=1.0,
        dice_weight=1.0,
        focal_weight=0.5,
        focal_gamma=2.0,
        label_smoothing=0.05,
        class_weights=None,
        lovasz_weight=0.0,
        rare_tversky_weight=0.0,
        rare_tversky_class=7,
        rare_tversky_alpha=0.3,
        rare_tversky_beta=0.7,
        rare_margin_weight=0.0,
        rare_margin_class=7,
        rare_margin=0.5,
        rare_margin_logit_clip=20.0,
        rare_confusion_weight=0.0,
        rare_confusion_target_class=7,
        rare_confusion_negative_classes=(1, 3),
        rare_confusion_margin=0.35,
        rare_confusion_topk=2,
        rare_confusion_max_violation=6.0,
        deep_supervision_aux_weight=0.4,
    ):
        super().__init__()

        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.lovasz_weight = lovasz_weight
        self.rare_tversky_weight = rare_tversky_weight
        self.rare_margin_weight = rare_margin_weight
        self.rare_confusion_weight = rare_confusion_weight
        self.deep_supervision_aux_weight = deep_supervision_aux_weight

        self.rare_margin = RareClassMarginLoss(
            target_class=rare_margin_class,
            margin=rare_margin,
            logit_clip=rare_margin_logit_clip,
            ignore_index=ignore_index,
        )

        self.rare_confusion = RareConfusionMarginLoss(
            target_class=rare_confusion_target_class,
            negative_classes=rare_confusion_negative_classes,
            margin=rare_confusion_margin,
            topk=rare_confusion_topk,
            max_violation=rare_confusion_max_violation,
            ignore_index=ignore_index,
        )

        

        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                self.class_weights = class_weights.detach().clone().float()
            else:
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

        # Use gentler sqrt weights for Dice to avoid over-penalizing major classes.
        if self.class_weights is not None:
            dice_weights = torch.sqrt(self.class_weights.detach().clone())
        else:
            dice_weights = None

        self.dice = DiceLoss(
            class_weights=dice_weights,
            ignore_index=ignore_index,
        )

        self.rare_tversky = RareClassTverskyLoss(
            target_class=rare_tversky_class,
            alpha=rare_tversky_alpha,
            beta=rare_tversky_beta,
            ignore_index=ignore_index,
        )

    def _ce_loss(self, logits, target):
        return F.cross_entropy(
            logits,
            target,
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

    def _focal_loss(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            ignore_index=self.ignore_index,
        )

        valid_mask = (target != self.ignore_index).float()

        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.focal_gamma) * ce
        focal = focal * valid_mask

        return focal.sum() / valid_mask.sum().clamp_min(1.0)

    def _lovasz_loss(self, logits, target):
        return lovasz_softmax(
            logits,
            target,
            ignore_index=self.ignore_index,
            classes="present",
        )

    def _single_loss(self, logits, target):
        valid_mask = target != self.ignore_index
        # Class-drop setting may create full-ignore samples; return zero loss safely.
        if not valid_mask.any():
            return logits.sum() * 0.0

        ce = self._ce_loss(logits, target)
        dice = self.dice(logits, target)

        focal = self._focal_loss(logits, target)

        loss = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.focal_weight * focal
        )

        if self.lovasz_weight > 0:
            lovasz = self._lovasz_loss(logits, target)
            loss = loss + self.lovasz_weight * lovasz

        if self.rare_tversky_weight > 0:
            loss = loss + self.rare_tversky_weight * self.rare_tversky(logits, target)

        if self.rare_margin_weight > 0:
            loss = loss + self.rare_margin_weight * self.rare_margin(logits, target)

        if self.rare_confusion_weight > 0:
            loss = loss + self.rare_confusion_weight * self.rare_confusion(logits, target)

        return loss

    def forward(self, pred, target):
        if isinstance(pred, dict):
            loss = self._single_loss(pred["main"], target)

            if "aux" in pred:
                aux_list = pred["aux"]
                if len(aux_list) > 0:
                    aux_weights = [
                        self.deep_supervision_aux_weight * (0.7 ** i)
                        for i in range(len(aux_list))
                    ]
                    for aux, w in zip(aux_list, aux_weights):
                        loss = loss + w * self._single_loss(aux, target)

            return loss

        return self._single_loss(pred, target)


# =========================
# Metric
# =========================
class SegMetric:
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        ignore_absent_classes: bool = False,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ignore_absent_classes = bool(ignore_absent_classes)
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)

        pred = pred.view(-1).cpu()
        target = target.view(-1).cpu()
        mask = (target >= 0) & (target < self.num_classes) & (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]

        inds = self.num_classes * target + pred
        cm = torch.bincount(inds, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += cm

    def get_scores(self):
        cm = self.confusion_matrix
        total = cm.sum()
        po = torch.diag(cm).sum() / (total + 1e-8)

        row_sum = cm.sum(dim=1)
        col_sum = cm.sum(dim=0)
        pe = (row_sum * col_sum).sum() / ((total * total) + 1e-8)
        kappa = (po - pe) / (1 - pe + 1e-8)

        iou = torch.diag(cm) / (row_sum + col_sum - torch.diag(cm) + 1e-8)
        f1 = 2 * torch.diag(cm) / (row_sum + col_sum + 1e-8)

        iou_all = iou.mean()
        f1_all = f1.mean()
        if self.ignore_absent_classes:
            present = row_sum > 0
            if present.any():
                iou_used = iou[present].mean()
                f1_used = f1[present].mean()
            else:
                iou_used = iou_all
                f1_used = f1_all
        else:
            iou_used = iou_all
            f1_used = f1_all

        return {
            "mIoU": float(iou_used),
            "OA": float(po),
            "Kappa": float(kappa),
            "mF1": float(f1_used),
            "mIoU_all": float(iou_all),
            "mF1_all": float(f1_all),
            "IoU_per_class": iou.tolist(),
            "F1_per_class": f1.tolist(),
        }


# =========================
# EMA
# =========================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.detach().clone()
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def apply_shadow(self, model):
        """
        鍏煎鏃ц缁冭剼鏈細
        鍏堝浠藉綋鍓嶅弬鏁帮紝鍐嶆妸 EMA 鍙傛暟鍐欏叆妯″瀷
        """
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Keep backup on CPU to avoid doubling GPU memory during validation.
                self.backup[name] = param.data.detach().cpu().clone()
                param.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model):
        """
        涓?apply_shadow 閰嶅浣跨敤锛?        楠岃瘉鍚庢仮澶嶅師濮嬭缁冨弬鏁?        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                backup = self.backup[name]
                if backup.device != param.data.device:
                    backup = backup.to(device=param.data.device, dtype=param.data.dtype, non_blocking=True)
                param.data.copy_(backup)
        self.backup = {}


EMA = ModelEMA


# =========================
# Optimizer / Scheduler / AMP builders
# =========================
def build_optimizer(cfg, model):
    base_lr = float(cfg.lr)
    backbone_lr = base_lr * float(getattr(cfg, "backbone_lr_mult", 1.0))
    weight_decay = float(getattr(cfg, "weight_decay", 1e-4))

    backbone_prefixes = ("opt_encoder.", "sar_encoder.")
    backbone_named = []
    head_named = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(backbone_prefixes):
            backbone_named.append((name, param))
        else:
            head_named.append((name, param))

    def split_decay(named_params):
        decay, no_decay = [], []
        for name, param in named_params:
            lname = name.lower()
            if (
                param.ndim <= 1
                or name.endswith(".bias")
                or "bn" in lname
                or "norm" in lname
            ):
                no_decay.append(param)
            else:
                decay.append(param)
        return decay, no_decay

    bb_decay, bb_no_decay = split_decay(backbone_named)
    hd_decay, hd_no_decay = split_decay(head_named)

    param_groups = []
    if bb_decay:
        param_groups.append({"params": bb_decay, "lr": backbone_lr, "weight_decay": weight_decay})
    if bb_no_decay:
        param_groups.append({"params": bb_no_decay, "lr": backbone_lr, "weight_decay": 0.0})
    if hd_decay:
        param_groups.append({"params": hd_decay, "lr": base_lr, "weight_decay": weight_decay})
    if hd_no_decay:
        param_groups.append({"params": hd_no_decay, "lr": base_lr, "weight_decay": 0.0})

    if not param_groups:
        raise RuntimeError("No trainable parameters found while building optimizer.")

    return torch.optim.AdamW(param_groups, lr=base_lr)



class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer,
        warmup_epochs=None,
        max_epochs=None,
        warmup_steps=None,
        max_steps=None,
        steps_per_epoch=1,
        min_lr=1e-6,
    ):
        self.optimizer = optimizer
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.min_lr = min_lr
        self.global_step = 0

        if warmup_steps is None:
            warmup_epochs = 0 if warmup_epochs is None else int(warmup_epochs)
            warmup_steps = warmup_epochs * self.steps_per_epoch
        if max_steps is None:
            if max_epochs is None:
                raise ValueError("Either max_steps or max_epochs must be provided.")
            max_steps = int(max_epochs) * self.steps_per_epoch

        self.warmup_steps = max(0, int(warmup_steps))
        self.max_steps = max(1, int(max_steps))
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        if self.warmup_steps > 0:
            for group in self.optimizer.param_groups:
                group["lr"] = self.min_lr

    def _compute_lrs(self, step):
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            factor = float(step) / float(self.warmup_steps)
            lrs = [
                self.min_lr + (base_lr - self.min_lr) * factor
                for base_lr in self.base_lrs
            ]
        else:
            denom = max(1, self.max_steps - self.warmup_steps)
            progress = (step - self.warmup_steps) / denom
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lrs = [
                self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs
            ]
        return lrs

    def step(self, step_idx=None):
        if step_idx is None:
            self.global_step += 1
        else:
            self.global_step = max(int(step_idx), 1)

        lrs = self._compute_lrs(self.global_step)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

def build_criterion(cfg, class_weights=None):
    neg_classes = getattr(cfg, "rare_confusion_negative_classes", (1, 3))
    if isinstance(neg_classes, int):
        neg_classes = (neg_classes,)
    elif isinstance(neg_classes, list):
        neg_classes = tuple(neg_classes)

    return SegLoss(
        ignore_index=cfg.ignore_index,
        ce_weight=getattr(cfg, "ce_weight", 1.0),
        dice_weight=getattr(cfg, "dice_weight", 1.0),
        focal_weight=getattr(cfg, "focal_weight", 0.5),
        focal_gamma=getattr(cfg, "focal_gamma", 2.0),
        label_smoothing=getattr(cfg, "label_smoothing", 0.05),
        class_weights=class_weights,
        lovasz_weight=getattr(cfg, "lovasz_weight", 0.0),
        rare_tversky_weight=getattr(cfg, "rare_tversky_weight", 0.0),
        rare_tversky_class=getattr(cfg, "rare_tversky_class", 7),
        rare_tversky_alpha=getattr(cfg, "rare_tversky_alpha", 0.3),
        rare_tversky_beta=getattr(cfg, "rare_tversky_beta", 0.7),
        rare_margin_weight=getattr(cfg, "rare_margin_weight", 0.0),
        rare_margin_class=getattr(cfg, "rare_margin_class", 7),
        rare_margin=getattr(cfg, "rare_margin", 0.5),
        rare_margin_logit_clip=getattr(cfg, "rare_margin_logit_clip", 20.0),
        rare_confusion_weight=getattr(cfg, "rare_confusion_weight", 0.0),
        rare_confusion_target_class=getattr(cfg, "rare_confusion_target_class", 7),
        rare_confusion_negative_classes=neg_classes,
        rare_confusion_margin=getattr(cfg, "rare_confusion_margin", 0.35),
        rare_confusion_topk=getattr(cfg, "rare_confusion_topk", 2),
        rare_confusion_max_violation=getattr(cfg, "rare_confusion_max_violation", 6.0),
        deep_supervision_aux_weight=getattr(cfg, "deep_supervision_aux_weight", 0.4),
    )


def build_scaler(cfg, device):
    use_amp = getattr(cfg, "use_amp", True) and device.type == "cuda"
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def build_ema(cfg, model):
    use_ema = getattr(cfg, "use_ema", True)
    if not use_ema:
        return None
    decay = getattr(cfg, "ema_decay", 0.999)
    return ModelEMA(model, decay=decay)



@torch.no_grad()
def compute_class_weights_from_loader(loader, num_classes, ignore_index=255, device="cpu"):
    hist = torch.zeros(num_classes, dtype=torch.float64)

    for batch in loader:
        y = batch["mask"]
        y = y.view(-1)
        y = y[(y >= 0) & (y < num_classes) & (y != ignore_index)]
        if y.numel() > 0:
            hist += torch.bincount(y, minlength=num_classes).double()

    freq = hist / hist.sum().clamp_min(1.0)
    weights = 1.0 / torch.log(1.02 + freq)
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights.to(device=device, dtype=torch.float32)
