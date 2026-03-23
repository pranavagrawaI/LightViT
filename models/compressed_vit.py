"""Compressed LightViT model definition with low-rank attention placeholders."""

from __future__ import annotations

import torch
from torch import nn

from .baseline_vit import LightViTBaseline
from .layers import LowRankLinear


class CompressedLightViT(LightViTBaseline):
    def __init__(self, rank_ratio: float = 0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < rank_ratio <= 1.0:
            raise ValueError("rank_ratio must be in the interval (0, 1].")
        self.rank_ratio = rank_ratio

    def apply_low_rank_head(self) -> None:
        input_dim = self.head.in_features
        output_dim = self.head.out_features
        rank = max(1, int(min(input_dim, output_dim) * self.rank_ratio))
        self.head = LowRankLinear(input_dim, output_dim, rank)




    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if isinstance(self.head, nn.Linear):
            self.apply_low_rank_head()
        return super().forward(images)