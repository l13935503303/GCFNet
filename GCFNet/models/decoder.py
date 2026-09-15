import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        values_per_channel = x.numel() // max(int(x.shape[1]), 1)
        if self.training and values_per_channel <= 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            SafeBatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class GLVRMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local_branch = nn.Sequential(
            ConvBNAct(channels, channels, k=3),
            ConvBNAct(channels, channels, k=1, p=0),
        )
        self.global_branch = nn.Sequential(
            ConvBNAct(channels, channels, k=1, p=0),
            ConvBNAct(channels, channels, k=3),
        )
        self.fuse = ConvBNAct(channels * 2, channels, k=1, p=0)

    def forward(self, x):
        xl = self.local_branch(x)
        xg = self.global_branch(x)
        return self.fuse(torch.cat([xl, xg], dim=1)) + x


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, k=3),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.block(x) + x)


class ASPPContext(nn.Module):
    def __init__(self, channels, dilations=(1, 3, 6), dropout=0.05):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBNAct(channels, channels, k=3, p=d, s=1)
            if d > 1 else ConvBNAct(channels, channels, k=1, p=0, s=1)
            for d in dilations
        ])
        for branch, dilation in zip(self.branches, dilations):
            conv = branch.block[0]
            conv.dilation = (int(dilation), int(dilation))
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNAct(channels, channels, k=1, p=0),
        )
        self.project = ConvBNAct(channels * (len(self.branches) + 1), channels, k=1, p=0)
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x):
        pooled = F.interpolate(self.image_pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        out = self.project(torch.cat([branch(x) for branch in self.branches] + [pooled], dim=1))
        return self.dropout(out) + x


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_glvrm=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False)
        block_cls = GLVRMBlock if bool(use_glvrm) else ResidualConvBlock
        self.block = nn.Sequential(block_cls(out_channels), block_cls(out_channels))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        skip = self.skip_proj(skip)
        return self.block(x + skip)
