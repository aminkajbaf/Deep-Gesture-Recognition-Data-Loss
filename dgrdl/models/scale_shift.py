"""
Scale-Shift MLP module (paper Sec. III-C).

Class-conditioned affine modulation of the encoder latent:
  z'_c = gamma_c * z + beta_c

Training  : use the ground-truth class to form a single transformed latent.
Inference : evaluate all class hypotheses and select the one with highest
            classifier confidence (max softmax probability).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleShiftMLP(nn.Module):
    """Maps a class index to (scale, shift) vectors of size latent_dim."""

    def __init__(self, latent_dim: int, num_classes: int, hidden: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.class_embed = nn.Embedding(num_classes, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SELU(inplace=True),
            nn.Linear(hidden, 2 * latent_dim),
        )
        # Identity-friendly init: scale ~ 1, shift ~ 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        with torch.no_grad():
            self.mlp[-1].bias[:latent_dim].fill_(1.0)

    def forward(self, class_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.class_embed(class_ids)
        params = self.mlp(h)
        scale, shift = params.chunk(2, dim=-1)
        return scale, shift

    def modulate(self, z: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        scale, shift = self.forward(class_ids)
        return scale * z + shift


class ScaleShiftClassifier(nn.Module):
    """Scale-Shift + fully-connected classifier with confidence selection."""

    def __init__(self, latent_dim: int, num_classes: int, hidden: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.scale_shift = ScaleShiftMLP(latent_dim, num_classes, hidden=hidden)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(
        self,
        z: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) encoder bottleneck flattened
            labels: optional ground-truth class ids (used in training)

        Returns:
            logits: (B, num_classes)
        """
        if self.training and labels is not None:
            z_mod = self.scale_shift.modulate(z, labels)
            return self.head(z_mod)

        # Inference / label-free: try all class hypotheses, pick highest confidence
        b = z.size(0)
        device = z.device
        best_logits = None
        best_conf = None
        for c in range(self.num_classes):
            class_ids = torch.full((b,), c, dtype=torch.long, device=device)
            z_mod = self.scale_shift.modulate(z, class_ids)
            logits = self.head(z_mod)
            conf = F.softmax(logits, dim=-1).max(dim=-1).values
            if best_logits is None:
                best_logits = logits
                best_conf = conf
            else:
                mask = conf > best_conf
                best_conf = torch.where(mask, conf, best_conf)
                best_logits = torch.where(mask.unsqueeze(-1), logits, best_logits)
        return best_logits
