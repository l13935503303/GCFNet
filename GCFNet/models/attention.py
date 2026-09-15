import torch
import torch.nn as nn


class GAMBlock(nn.Module):
    def __init__(self, channels, reduction=4, use_residual=True):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.use_residual = use_residual

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        identity = x
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        if self.use_residual:
            x = x + identity
        return x