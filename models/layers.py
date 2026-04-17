"""Custom PyTorch layers used by LightViT experiments."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from core.factorization import factorize_linear_weight, factorize_linear_weight_act_svd


class LowRankLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int, bias: bool = True
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.left = nn.Linear(in_features, rank, bias=False)
        self.right = nn.Linear(rank, out_features, bias=bias)

    @classmethod
    def from_weight(
        cls, weight: torch.Tensor, bias: torch.Tensor | None, rank: int
    ) -> "LowRankLinear":
        rank = min(rank, min(weight.shape))
        module = cls(weight.shape[1], weight.shape[0], rank, bias=bias is not None)
        left_rank, right_rank = factorize_linear_weight(weight.detach(), rank)
        with torch.no_grad():
            module.left.weight.copy_(right_rank)
            module.right.weight.copy_(left_rank)
            if bias is not None:
                module.right.bias.copy_(bias.detach())
        return module

    @classmethod
    def from_activation_aware_weight(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        covariance: torch.Tensor,
        rank: int,
    ) -> "LowRankLinear":
        rank = min(rank, min(weight.shape))
        module = cls(weight.shape[1], weight.shape[0], rank, bias=bias is not None)
        left_rank, right_rank = factorize_linear_weight_act_svd(
            weight.detach(), covariance, rank
        )
        with torch.no_grad():
            module.left.weight.copy_(right_rank)
            module.right.weight.copy_(left_rank)
            if bias is not None:
                module.right.bias.copy_(bias.detach())
        return module

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int) -> "LowRankLinear":
        return cls.from_weight(linear.weight, linear.bias, rank)

    @classmethod
    def from_linear_activation_aware(
        cls, linear: nn.Linear, covariance: torch.Tensor, rank: int
    ) -> "LowRankLinear":
        return cls.from_activation_aware_weight(
            linear.weight, linear.bias, covariance, rank
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(inputs))


class LowRankSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rank: int,
        dropout: float = 0.0,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self._qkv_same_embed_dim = True
        self.q_proj = LowRankLinear(embed_dim, embed_dim, rank)
        self.k_proj = LowRankLinear(embed_dim, embed_dim, rank)
        self.v_proj = LowRankLinear(embed_dim, embed_dim, rank)
        self.out_proj = LowRankLinear(embed_dim, embed_dim, rank)

    @property
    def in_proj_bias(self) -> None:
        return None

    @property
    def in_proj_weight(self) -> None:
        return None

    @classmethod
    def from_multihead_attention(
        cls, attention: nn.MultiheadAttention, rank: int
    ) -> "LowRankSelfAttention":
        module = cls(
            embed_dim=attention.embed_dim,
            num_heads=attention.num_heads,
            rank=rank,
            dropout=attention.dropout,
            batch_first=attention.batch_first,
        )
        if attention.in_proj_weight is None:
            raise ValueError("Expected packed qkv weights.")
        q_weight, k_weight, v_weight = attention.in_proj_weight.detach().chunk(3, dim=0)
        if attention.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = attention.in_proj_bias.detach().chunk(3, dim=0)

        module.q_proj = LowRankLinear.from_weight(q_weight, q_bias, rank)
        module.k_proj = LowRankLinear.from_weight(k_weight, k_bias, rank)
        module.v_proj = LowRankLinear.from_weight(v_weight, v_bias, rank)
        module.out_proj = LowRankLinear.from_linear(attention.out_proj, rank)
        return module

    @classmethod
    def from_multihead_attention_activation_aware(
        cls,
        attention: nn.MultiheadAttention,
        qkv_covariance: torch.Tensor,
        rank: int,
        out_proj_covariance: torch.Tensor | None = None,
    ) -> "LowRankSelfAttention":
        module = cls(
            embed_dim=attention.embed_dim,
            num_heads=attention.num_heads,
            rank=rank,
            dropout=attention.dropout,
            batch_first=attention.batch_first,
        )
        if attention.in_proj_weight is None:
            raise ValueError("Expected packed qkv weights.")
        q_weight, k_weight, v_weight = attention.in_proj_weight.detach().chunk(3, dim=0)
        if attention.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = attention.in_proj_bias.detach().chunk(3, dim=0)

        module.q_proj = LowRankLinear.from_activation_aware_weight(
            q_weight, q_bias, qkv_covariance, rank
        )
        module.k_proj = LowRankLinear.from_activation_aware_weight(
            k_weight, k_bias, qkv_covariance, rank
        )
        module.v_proj = LowRankLinear.from_activation_aware_weight(
            v_weight, v_bias, qkv_covariance, rank
        )
        if out_proj_covariance is None:
            module.out_proj = LowRankLinear.from_linear(attention.out_proj, rank)
        else:
            module.out_proj = LowRankLinear.from_linear_activation_aware(
                attention.out_proj, out_proj_covariance, rank
            )
        return module

    def _shape_projection(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = inputs.shape
        return inputs.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _additive_mask(
        self,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        query: torch.Tensor,
        source_len: int,
    ) -> torch.Tensor | None:
        mask = None
        batch_size, target_len, _ = query.shape

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.zeros(
                    attn_mask.shape, dtype=query.dtype, device=query.device
                ).masked_fill(attn_mask, float("-inf"))
            else:
                attn_mask = attn_mask.to(dtype=query.dtype, device=query.device)

            if attn_mask.ndim == 2:
                mask = attn_mask.view(1, 1, target_len, source_len)
            elif (
                attn_mask.ndim == 3
                and attn_mask.shape[0] == batch_size * self.num_heads
            ):
                mask = attn_mask.view(
                    batch_size, self.num_heads, target_len, source_len
                )
            elif attn_mask.ndim == 3:
                mask = attn_mask.unsqueeze(1)
            else:
                mask = attn_mask

        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                padding_mask = torch.zeros(
                    key_padding_mask.shape, dtype=query.dtype, device=query.device
                ).masked_fill(key_padding_mask, float("-inf"))
            else:
                padding_mask = key_padding_mask.to(dtype=query.dtype, device=query.device)
            padding_mask = padding_mask.view(batch_size, 1, 1, source_len)
            mask = padding_mask if mask is None else mask + padding_mask

        return mask

    def _attention_with_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        is_causal: bool,
        average_attn_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        if is_causal:
            target_len, source_len = scores.shape[-2:]
            causal_mask = torch.ones(
                target_len, source_len, dtype=torch.bool, device=scores.device
            ).triu(1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        outputs = weights @ v
        if average_attn_weights:
            weights = weights.mean(dim=1)
        return outputs, weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.batch_first:
            query, key, value = (
                tensor.transpose(0, 1) for tensor in (query, key, value)
            )

        q = self._shape_projection(self.q_proj(query))
        k = self._shape_projection(self.k_proj(key))
        v = self._shape_projection(self.v_proj(value))
        mask = self._additive_mask(attn_mask, key_padding_mask, query, key.shape[1])

        if need_weights:
            outputs, weights = self._attention_with_weights(
                q, k, v, mask, is_causal, average_attn_weights
            )
        else:
            outputs = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            weights = None

        outputs = outputs.transpose(1, 2).contiguous().view(
            query.shape[0], query.shape[1], self.embed_dim
        )
        outputs = self.out_proj(outputs)
        if not self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, weights


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
            raise ValueError(
                "TokenMerging expects input of shape [batch, tokens, channels]."
            )
        return tokens[:, :: self.stride, :]
