"""
Bi-oriented Spatial Mamba (BSMamba) encoder-decoder for DGRDL.

Architecture (paper Sec. III):
  Input  : (B, 20, 32, 32)
  Stem   : depthwise 3x3 + pointwise 20 -> 32
  Encoder: 4 BSMamba stages with downsampling -> bottleneck 512 @ 2x2
  Decoder: ConvTranspose ups with multiplicative skip connections
  Recon  : 1x1 conv mapping decoder features back to 20 channels
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mamba-ssm is required for BSMamba. Install with: pip install mamba-ssm"
    ) from exc

from .scale_shift import ScaleShiftClassifier

BlockType = Literal["mamba", "transformer"]


class Stem(nn.Module):
    def __init__(self, in_ch: int = 20, out_ch: int = 32):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class Downsample2D(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BSMambaCore(nn.Module):
    """Bidirectional 1D Mamba along flattened spatial tokens."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mf = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mb = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.norm(seq)
        yf = self.mf(h)
        yb = torch.flip(self.mb(torch.flip(h, dims=[1])), dims=[1])
        return seq + yf + yb


class ManualMHSA(nn.Module):
    def __init__(self, dim: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        if dim % nhead != 0:
            raise ValueError("dim must be divisible by nhead")
        self.nhead = nhead
        self.dh = dim // nhead
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.dh**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.nhead, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.dropout((q * self.scale @ k.transpose(-2, -1)).softmax(dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(b, l, d)
        return self.proj(y)


class TransformerAttnFFN(nn.Module):
    def __init__(self, dim: int, nhead: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ManualMHSA(dim, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = max(int(dim * mlp_ratio), dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        seq = seq + self.attn(self.norm1(seq))
        return seq + self.mlp(self.norm2(seq))


class TransformerStackCore(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        depth: int,
        mlp_ratio: float,
        dropout: float = 0.0,
        weight_tied_unroll: int = 1,
    ):
        super().__init__()
        self.weight_tied_unroll = max(1, weight_tied_unroll)
        self.layers = nn.ModuleList(
            [TransformerAttnFFN(dim, nhead, mlp_ratio, dropout) for _ in range(depth)]
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        for _ in range(self.weight_tied_unroll):
            for layer in self.layers:
                seq = layer(seq)
        return seq


class StageBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        block: BlockType,
        *,
        nhead: int = 4,
        stage_ff_ratio: float = 2.0,
        mamba_expand: int = 2,
        mamba_d_state: int = 16,
        transformer_stack_depth: int = 1,
        transformer_attn_mlp_ratio: float = 4.0,
        transformer_weight_tied_unroll: int = 1,
    ):
        super().__init__()
        if block == "mamba":
            self.core = BSMambaCore(dim, d_state=mamba_d_state, expand=mamba_expand)
        else:
            self.core = TransformerStackCore(
                dim,
                nhead,
                depth=transformer_stack_depth,
                mlp_ratio=transformer_attn_mlp_ratio,
                weight_tied_unroll=transformer_weight_tied_unroll,
            )
        self.norm_m = nn.LayerNorm(dim)
        hidden = max(int(dim * stage_ff_ratio), dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SELU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        seq = self.core(seq)
        seq = seq + self.ff(self.norm_m(seq))
        return seq.transpose(1, 2).reshape(b, c, h, w)


def _nhead_for(dim: int) -> int:
    nh = max(1, dim // 16)
    if dim % nh != 0:
        nh = min(n for n in (1, 2, 4, 8) if dim % n == 0)
    return nh


class BSMambaEncoderDecoder(nn.Module):
    """Encoder-decoder backbone with multiplicative skips."""

    def __init__(
        self,
        in_channels: int = 20,
        depth_per_stage: int = 2,
        block: BlockType = "mamba",
        bottleneck_depth: int = 2,
        mamba_expand: int = 2,
        mamba_d_state: int = 16,
        transformer_stack_depth: int = 1,
        transformer_attn_mlp_ratio: float = 4.0,
        transformer_weight_tied_unroll: int = 1,
        stage_ff_ratio: float = 2.0,
    ):
        super().__init__()
        self.stem = Stem(in_channels, 32)
        chs = [32, 64, 128, 256, 512]
        self.downs = nn.ModuleList([Downsample2D(chs[i], chs[i + 1]) for i in range(4)])
        self.stages = nn.ModuleList()
        for s in range(4):
            dim = chs[s]
            self.stages.append(
                nn.ModuleList(
                    [
                        StageBlock(
                            dim,
                            block,
                            nhead=_nhead_for(dim),
                            stage_ff_ratio=stage_ff_ratio,
                            mamba_expand=mamba_expand,
                            mamba_d_state=mamba_d_state,
                            transformer_stack_depth=transformer_stack_depth,
                            transformer_attn_mlp_ratio=transformer_attn_mlp_ratio,
                            transformer_weight_tied_unroll=transformer_weight_tied_unroll,
                        )
                        for _ in range(depth_per_stage)
                    ]
                )
            )
        self.bottleneck = nn.ModuleList(
            [
                StageBlock(
                    512,
                    block,
                    nhead=_nhead_for(512),
                    stage_ff_ratio=stage_ff_ratio,
                    mamba_expand=mamba_expand,
                    mamba_d_state=mamba_d_state,
                    transformer_stack_depth=transformer_stack_depth,
                    transformer_attn_mlp_ratio=transformer_attn_mlp_ratio,
                    transformer_weight_tied_unroll=transformer_weight_tied_unroll,
                )
                for _ in range(bottleneck_depth)
            ]
        )
        self.up_convs = nn.ModuleList(
            [
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            ]
        )
        self.skip_projs = nn.ModuleList(
            [
                nn.Conv2d(256, 256, 1, bias=False),
                nn.Conv2d(128, 128, 1, bias=False),
                nn.Conv2d(64, 64, 1, bias=False),
                nn.Conv2d(32, 32, 1, bias=False),
            ]
        )
        self.dec_bn = nn.ModuleList([nn.BatchNorm2d(c) for c in (256, 128, 64, 32)])
        self.recon_head = nn.Conv2d(32, in_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        skips: List[torch.Tensor] = []
        for s in range(4):
            for blk in self.stages[s]:
                x = blk(x)
            skips.append(x)
            x = self.downs[s](x)
        for blk in self.bottleneck:
            x = blk(x)
        return x, skips

    def decode(self, latent: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        x = latent
        for i, up in enumerate(self.up_convs):
            x = up(x)
            sk = self.skip_projs[i](skips[3 - i])
            x = self.dec_bn[i](x * F.selu(sk))
        return x

    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, bottleneck_latent)."""
        latent, skips = self.encode(x)
        feat = self.decode(latent, skips)
        recon = self.recon_head(feat)
        return recon, latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.reconstruct(x)
        return recon


class DGRDLModel(nn.Module):
    """
    Full DGRDL system:
      Phase 1 uses reconstruct()
      Phase 2 / inference uses classify() with Scale-Shift
    """

    def __init__(
        self,
        num_classes: int = 13,
        in_channels: int = 20,
        depth_per_stage: int = 2,
        block: BlockType = "mamba",
        scale_shift_hidden: int = 512,
        **backbone_kwargs,
    ):
        super().__init__()
        self.backbone = BSMambaEncoderDecoder(
            in_channels=in_channels,
            depth_per_stage=depth_per_stage,
            block=block,
            **backbone_kwargs,
        )
        self.latent_dim = 512 * 2 * 2
        self.classifier = ScaleShiftClassifier(
            latent_dim=self.latent_dim,
            num_classes=num_classes,
            hidden=scale_shift_hidden,
        )

    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.backbone.reconstruct(x)

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.backbone.encode(x)
        return latent.flatten(1)

    def classify(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.encode_latent(x)
        return self.classifier(z, labels=labels)

    def freeze_encoder(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = "classify",
    ):
        if mode == "reconstruct":
            return self.reconstruct(x)
        return self.classify(x, labels=labels)
