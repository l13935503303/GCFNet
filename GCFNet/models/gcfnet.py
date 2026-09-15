import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GAMBlock
from .decoder import ASPPContext, ConvBNAct, DecoderStage, GLVRMBlock
from .encoders import HRNetW48Encoder, ResNet101Encoder
from .fusion import GMF


class MeanFusion(nn.Module):
    def forward(self, opt: torch.Tensor, sar: torch.Tensor):
        return 0.5 * (opt + sar)


class GCFNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        opt_in_channels: int = 3,
        sar_in_channels: int = 1,
        encoder_dims=None,
        sar_encoder_dims=None,
        decoder_dim: int = 192,
        deep_supervision: bool = True,
        pretrained_backbone: bool = False,
        eca_kernel_size: int = 3,
        use_opt_modality: bool = True,
        use_sar_modality: bool = True,
        use_gmf_module: bool = True,
        gmf_use_residual_sum: bool = True,
        gam_reduction: int = 4,
        use_gam_module: bool = True,
        gam_use_residual: bool = True,
        opt_pretrained_path: str = "",
        sar_pretrained_path: str = "",
        use_glvrm: bool = True,
        glvrm_decoder_stages: int = 1,
        use_aspp_context: bool = True,
        aspp_dilations=(1, 3, 6),
        aspp_dropout: float = 0.05,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_opt_modality = bool(use_opt_modality)
        self.use_sar_modality = bool(use_sar_modality)
        self.use_gmf_module = bool(use_gmf_module)
        self.use_gam_module = bool(use_gam_module)
        if (not self.use_opt_modality) and (not self.use_sar_modality):
            raise ValueError("At least one modality must be enabled.")

        self.opt_encoder = HRNetW48Encoder(
            pretrained=pretrained_backbone,
            pretrained_path=opt_pretrained_path,
        )
        self.sar_encoder = ResNet101Encoder(
            in_ch=sar_in_channels,
            pretrained=pretrained_backbone,
            pretrained_path=sar_pretrained_path,
        )

        real_opt_dims = list(self.opt_encoder.out_channels)
        real_sar_dims = list(self.sar_encoder.out_channels)

        print(f"[INFO] Optical encoder channels: {real_opt_dims}")
        print(f"[INFO] SAR encoder channels: {real_sar_dims}")

        self.encoder_dims = tuple(real_opt_dims)
        self.sar_encoder_dims = tuple(real_sar_dims)

        if decoder_dim is None:
            decoder_dim = self.encoder_dims[-2]
        self.decoder_dim = decoder_dim

        # Align both modality distributions before gated fusion.
        self.opt_align = nn.ModuleList([
            ConvBNAct(in_c, out_c, k=1, p=0)
            for in_c, out_c in zip(self.encoder_dims, self.encoder_dims)
        ])
        self.sar_align = nn.ModuleList([
            ConvBNAct(in_c, out_c, k=1, p=0)
            for in_c, out_c in zip(self.sar_encoder_dims, self.encoder_dims)
        ])

        self.fusions = nn.ModuleList([
            GMF(c, eca_kernel_size=eca_kernel_size, use_residual_sum=gmf_use_residual_sum)
            if self.use_gmf_module else MeanFusion()
            for c in self.encoder_dims
        ])

        self.gams = nn.ModuleList([
            GAMBlock(c, reduction=gam_reduction, use_residual=gam_use_residual)
            if self.use_gam_module else nn.Identity()
            for c in self.encoder_dims
        ])

        neck_layers = [
            nn.Conv2d(self.encoder_dims[-1], self.decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.decoder_dim),
            nn.GELU(),
        ]
        if bool(use_aspp_context):
            neck_layers.append(ASPPContext(self.decoder_dim, dilations=aspp_dilations, dropout=aspp_dropout))
        if bool(use_glvrm):
            neck_layers.append(GLVRMBlock(self.decoder_dim))
        self.neck = nn.Sequential(*neck_layers)

        rev_dims = list(reversed(self.encoder_dims[:-1]))
        self.decoder_stages = nn.ModuleList([
            DecoderStage(
                self.decoder_dim,
                skip_c,
                self.decoder_dim,
                use_glvrm=bool(use_glvrm) and (i < int(glvrm_decoder_stages)),
            )
            for i, skip_c in enumerate(rev_dims)
        ])

        self.seg_head = nn.Sequential(
            ConvBNAct(self.decoder_dim, self.decoder_dim, 3),
            nn.Conv2d(self.decoder_dim, num_classes, kernel_size=1),
        )

        if self.deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(self.decoder_dim, num_classes, kernel_size=1)
                for _ in range(len(self.decoder_stages))
            ])

    def encode(self, x_opt: torch.Tensor, x_sar: torch.Tensor):
        opt_feats = self.opt_encoder(x_opt)
        sar_feats = self.sar_encoder(x_sar)

        fused_feats = []
        for fo, fs, opt_align, sar_align, fusion, gam in zip(
            opt_feats, sar_feats, self.opt_align, self.sar_align, self.fusions, self.gams
        ):
            fo = opt_align(fo)
            fs = sar_align(fs)

            if fs.shape[-2:] != fo.shape[-2:]:
                fs = F.interpolate(fs, size=fo.shape[-2:], mode="bilinear", align_corners=False)

            if not self.use_opt_modality:
                fo = torch.zeros_like(fo)
            if not self.use_sar_modality:
                fs = torch.zeros_like(fs)

            fused = fusion(fo, fs)
            fused = gam(fused)
            fused_feats.append(fused)

        return fused_feats

    def forward(self, x_opt: torch.Tensor, x_sar: torch.Tensor):
        input_size = x_opt.shape[-2:]
        feats = self.encode(x_opt, x_sar)
        x = self.neck(feats[-1])

        aux_outputs = []
        for i, (stage, skip) in enumerate(zip(self.decoder_stages, reversed(feats[:-1]))):
            x = stage(x, skip)
            if self.deep_supervision:
                aux = self.aux_heads[i](x)
                aux = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False)
                aux_outputs.append(aux)

        out = self.seg_head(x)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        if self.deep_supervision:
            return {"main": out, "aux": aux_outputs}
        return out
