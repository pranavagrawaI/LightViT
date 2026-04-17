"""Compressed LightViT model definition with low-rank attention placeholders."""

from __future__ import annotations

import torch
from torch import nn

from core.factorization import rank_from_ratio
from .baseline_vit import LightViTBaseline
from .layers import LowRankLinear, LowRankSelfAttention


class CompressedLightViT(LightViTBaseline):
    def __init__(self, rank_ratio: float = 0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < rank_ratio <= 1.0:
            raise ValueError("rank_ratio must be in the interval (0, 1].")
        self.rank_ratio = rank_ratio

    def apply_low_rank_head(self) -> None:
        if isinstance(self.head, LowRankLinear):
            return
        input_dim = self.head.in_features
        output_dim = self.head.out_features
        rank = max(1, int(min(input_dim, output_dim) * self.rank_ratio))
        self.head = LowRankLinear.from_linear(self.head, rank)

    def _rank_for_linear(self, linear: nn.Linear, rank_ratio: float) -> int:
        return rank_from_ratio(linear.weight, rank_ratio)

    def apply_pure_svd(self, rank_ratio: float | None = None) -> None:
        if rank_ratio is None:
            rank_ratio = self.rank_ratio
        if not 0.0 < rank_ratio <= 1.0:
            raise ValueError("rank_ratio must be in the interval (0, 1].")

        for block in self.blocks.layers:
            if isinstance(block.self_attn, nn.MultiheadAttention):
                rank = rank_from_ratio(block.self_attn.in_proj_weight, rank_ratio)
                block.self_attn = LowRankSelfAttention.from_multihead_attention(
                    block.self_attn, rank
                )

            if isinstance(block.linear1, nn.Linear):
                rank = self._rank_for_linear(block.linear1, rank_ratio)
                block.linear1 = LowRankLinear.from_linear(block.linear1, rank)

            if isinstance(block.linear2, nn.Linear):
                rank = self._rank_for_linear(block.linear2, rank_ratio)
                block.linear2 = LowRankLinear.from_linear(block.linear2, rank)

        if isinstance(self.head, nn.Linear):
            rank = self._rank_for_linear(self.head, rank_ratio)
            self.head = LowRankLinear.from_linear(self.head, rank)

    def _covariance_for(
        self,
        covariances: dict[str, torch.Tensor],
        name: str,
        expected_dim: int,
    ) -> torch.Tensor:
        covariance = covariances.get(name)
        if covariance is None:
            raise KeyError(f"Missing activation covariance for {name}.")
        if covariance.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Covariance for {name} must have shape "
                f"({expected_dim}, {expected_dim}), got {tuple(covariance.shape)}."
            )
        return covariance

    def apply_act_svd(
        self,
        covariances: dict[str, torch.Tensor],
        rank_ratio: float | None = None,
    ) -> None:
        if rank_ratio is None:
            rank_ratio = self.rank_ratio
        if not 0.0 < rank_ratio <= 1.0:
            raise ValueError("rank_ratio must be in the interval (0, 1].")

        for layer_idx, block in enumerate(self.blocks.layers):
            block_name = f"blocks.layers.{layer_idx}"
            if isinstance(block.self_attn, nn.MultiheadAttention):
                rank = rank_from_ratio(block.self_attn.in_proj_weight, rank_ratio)
                covariance = self._covariance_for(
                    covariances,
                    f"{block_name}.self_attn",
                    block.self_attn.embed_dim,
                )
                block.self_attn = (
                    LowRankSelfAttention.from_multihead_attention_activation_aware(
                        block.self_attn, covariance, rank
                    )
                )

            if isinstance(block.linear1, nn.Linear):
                rank = self._rank_for_linear(block.linear1, rank_ratio)
                covariance = self._covariance_for(
                    covariances, f"{block_name}.linear1", block.linear1.in_features
                )
                block.linear1 = LowRankLinear.from_linear_activation_aware(
                    block.linear1, covariance, rank
                )

            if isinstance(block.linear2, nn.Linear):
                rank = self._rank_for_linear(block.linear2, rank_ratio)
                covariance = self._covariance_for(
                    covariances, f"{block_name}.linear2", block.linear2.in_features
                )
                block.linear2 = LowRankLinear.from_linear_activation_aware(
                    block.linear2, covariance, rank
                )

        if isinstance(self.head, nn.Linear):
            rank = self._rank_for_linear(self.head, rank_ratio)
            covariance = self._covariance_for(covariances, "head", self.head.in_features)
            self.head = LowRankLinear.from_linear_activation_aware(
                self.head, covariance, rank
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return super().forward(images)
