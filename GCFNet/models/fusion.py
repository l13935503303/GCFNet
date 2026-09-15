import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)

    def forward(self, x):
        y = x.mean((2, 3), keepdim=True)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return torch.sigmoid(y)


class ChannelGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class GMF(nn.Module):
    def __init__(self, channels, eca_kernel_size=3, use_residual_sum=True):
        super().__init__()
        self.channels = channels
        self.use_residual_sum = use_residual_sum

        self.eca_opt = ECA(k_size=eca_kernel_size)
        self.eca_sar = ECA(k_size=eca_kernel_size)

        self.gate_opt = ChannelGate(channels)
        self.gate_sar = ChannelGate(channels)

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, opt, sar):
        b, c, h, w = opt.shape

        assert c == self.channels, (
            f"GMF expected {self.channels} channels, got {c}. "
            f"Opt feature shape: {tuple(opt.shape)}"
        )
        assert sar.shape[1] == c, (
            f"GMF channel mismatch: opt={c}, sar={sar.shape[1]}, "
            f"sar shape: {tuple(sar.shape)}"
        )
        assert sar.shape[-2:] == (h, w), (
            f"GMF spatial mismatch: opt={tuple(opt.shape)}, sar={tuple(sar.shape)}"
        )

        w_opt = self.eca_opt(opt).flatten(1)
        w_sar = self.eca_sar(sar).flatten(1)

        g_opt = self.gate_opt(w_opt).view(b, c, 1, 1)
        g_sar = self.gate_sar(w_sar).view(b, c, 1, 1)

        fused = g_opt * opt + g_sar * sar
        if self.use_residual_sum:
            fused = fused + 0.5 * (opt + sar)

        fused = self.refine(fused)
        return F.gelu(fused)
