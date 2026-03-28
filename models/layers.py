"""Custom PyTorch layers used by LightViT experiments."""

from __future__ import annotations

import torch
from torch import nn


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.left = nn.Linear(in_features, rank, bias=False)
        self.right = nn.Linear(rank, out_features, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(inputs))


class QATLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fake_quant = torch.ao.quantization.FakeQuantize()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quantized_inputs = self.fake_quant(inputs)
        outputs = self.linear(quantized_inputs)
        return self.fake_quant(outputs)


class TokenMerging(nn.Module):
    def __init__(self, stride: int = 2) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("TokenMerging expects input of shape [batch, tokens, channels].")
        return tokens[:, :: self.stride, :]