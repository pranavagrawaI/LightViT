"""Hybrid LightViT with Act-SVD MLPs and Tucker-factorized QKV attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from core.factorization import rank_from_ratio
from .baseline_vit import LightViTBaseline
from .layers import LowRankLinear


def _mode_product(tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
    result = torch.tensordot(matrix, tensor, dims=([1], [mode]))
    return torch.movedim(result, 0, mode)


def _mode_factor(tensor: torch.Tensor, mode: int, rank: int) -> torch.Tensor:
    unfolded = torch.movedim(tensor, mode, 0).reshape(tensor.shape[mode], -1)
    left, _singular_values, _right_t = torch.linalg.svd(unfolded, full_matrices=False)
    return left[:, :rank]


def _tucker_qkv(
    in_proj_weight: torch.Tensor,
    num_heads: int,
    head_rank: int,
    out_rank: int,
    in_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embed_dim = in_proj_weight.shape[1]
    head_dim = embed_dim // num_heads
    tensor = in_proj_weight.detach().view(3, num_heads, head_dim, embed_dim)

    factor_heads = _mode_factor(tensor, 1, head_rank)
    factor_out = _mode_factor(tensor, 2, out_rank)
    factor_in = _mode_factor(tensor, 3, in_rank)

    core = _mode_product(tensor, factor_heads.transpose(0, 1), 1)
    core = _mode_product(core, factor_out.transpose(0, 1), 2)
    core = _mode_product(core, factor_in.transpose(0, 1), 3)
    return core, factor_heads, factor_out, factor_in


def _tucker_core_from_factors(
    tensor: torch.Tensor,
    factor_heads: torch.Tensor,
    factor_out: torch.Tensor,
    factor_in: torch.Tensor,
) -> torch.Tensor:
    core = _mode_product(tensor, factor_heads.transpose(0, 1), 1)
    core = _mode_product(core, factor_out.transpose(0, 1), 2)
    core = _mode_product(core, factor_in.transpose(0, 1), 3)
    return core


def _hooi_update(
    tensor: torch.Tensor,
    factor_heads: torch.Tensor,
    factor_out: torch.Tensor,
    factor_in: torch.Tensor,
    mode: int,
    rank: int,
) -> torch.Tensor:
    projected = tensor
    if mode != 1:
        projected = _mode_product(projected, factor_heads.transpose(0, 1), 1)
    if mode != 2:
        projected = _mode_product(projected, factor_out.transpose(0, 1), 2)
    if mode != 3:
        projected = _mode_product(projected, factor_in.transpose(0, 1), 3)
    return _mode_factor(projected, mode, rank)


def _tucker_qkv_hooi(
    in_proj_weight: torch.Tensor,
    num_heads: int,
    head_rank: int,
    out_rank: int,
    in_rank: int,
    iterations: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embed_dim = in_proj_weight.shape[1]
    head_dim = embed_dim // num_heads
    tensor = in_proj_weight.detach().view(3, num_heads, head_dim, embed_dim)
    _core, factor_heads, factor_out, factor_in = _tucker_qkv(
        in_proj_weight, num_heads, head_rank, out_rank, in_rank
    )

    for _ in range(iterations):
        factor_heads = _hooi_update(
            tensor, factor_heads, factor_out, factor_in, 1, head_rank
        )
        factor_out = _hooi_update(
            tensor, factor_heads, factor_out, factor_in, 2, out_rank
        )
        factor_in = _hooi_update(
            tensor, factor_heads, factor_out, factor_in, 3, in_rank
        )

    core = _tucker_core_from_factors(tensor, factor_heads, factor_out, factor_in)
    return core, factor_heads, factor_out, factor_in


def _tucker_qkv_tensorly(
    in_proj_weight: torch.Tensor,
    num_heads: int,
    head_rank: int,
    out_rank: int,
    in_rank: int,
    iterations: int = 100,
    tol: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        import tensorly as tl
        from tensorly.decomposition import tucker
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Tensorly Tucker backend requested, but tensorly is not installed. "
            "Install project requirements first."
        ) from exc

    tl.set_backend("pytorch")
    embed_dim = in_proj_weight.shape[1]
    head_dim = embed_dim // num_heads
    tensor = in_proj_weight.detach().view(3, num_heads, head_dim, embed_dim)
    result = tucker(
        tensor,
        rank=[3, head_rank, out_rank, in_rank],
        init="svd",
        n_iter_max=iterations,
        tol=tol,
    )
    if hasattr(result, "core"):
        core = result.core
        factors = result.factors
    else:
        core, factors = result

    factor_qkv, factor_heads, factor_out, factor_in = factors
    core = _mode_product(core, factor_qkv, 0)
    return core, factor_heads, factor_out, factor_in


def tucker_qkv_relative_error(
    in_proj_weight: torch.Tensor,
    num_heads: int,
    core: torch.Tensor,
    factor_heads: torch.Tensor,
    factor_out: torch.Tensor,
    factor_in: torch.Tensor,
) -> float:
    reconstructed = _mode_product(core, factor_heads, 1)
    reconstructed = _mode_product(reconstructed, factor_out, 2)
    reconstructed = _mode_product(reconstructed, factor_in, 3)
    reconstructed = reconstructed.reshape_as(in_proj_weight)
    denominator = in_proj_weight.norm().clamp_min(1e-12)
    return float((in_proj_weight - reconstructed).norm() / denominator)


class TuckerSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_rank: int,
        out_rank: int,
        in_rank: int,
        dropout: float = 0.0,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.head_rank = head_rank
        self.out_rank = out_rank
        self.in_rank = in_rank
        self.dropout = dropout
        self.batch_first = batch_first
        self._qkv_same_embed_dim = True

        self.core = nn.Parameter(torch.empty(3, head_rank, out_rank, in_rank))
        self.factor_heads = nn.Parameter(torch.empty(num_heads, head_rank))
        self.factor_out = nn.Parameter(torch.empty(self.head_dim, out_rank))
        self.factor_in = nn.Parameter(torch.empty(embed_dim, in_rank))
        self.qkv_bias = nn.Parameter(torch.zeros(3, num_heads, self.head_dim))
        self.out_proj = LowRankLinear(embed_dim, embed_dim, in_rank)

    @property
    def in_proj_bias(self) -> None:
        return None

    @property
    def in_proj_weight(self) -> None:
        return None

    @classmethod
    def from_multihead_attention(
        cls,
        attention: nn.MultiheadAttention,
        rank_ratio: float,
        head_rank: int | None = None,
        tucker_method: str = "hosvd",
        hooi_iterations: int = 8,
        tensorly_tol: float = 1e-5,
    ) -> "TuckerSelfAttention":
        if attention.in_proj_weight is None:
            raise ValueError("Expected packed qkv weights.")
        embed_dim = attention.embed_dim
        head_dim = embed_dim // attention.num_heads
        head_rank = attention.num_heads if head_rank is None else head_rank
        out_rank = max(1, int(head_dim * rank_ratio))
        in_rank = max(1, int(embed_dim * rank_ratio))

        module = cls(
            embed_dim=embed_dim,
            num_heads=attention.num_heads,
            head_rank=head_rank,
            out_rank=out_rank,
            in_rank=in_rank,
            dropout=attention.dropout,
            batch_first=attention.batch_first,
        )
        if tucker_method == "hosvd":
            core, factor_heads, factor_out, factor_in = _tucker_qkv(
                attention.in_proj_weight,
                attention.num_heads,
                head_rank,
                out_rank,
                in_rank,
            )
        elif tucker_method == "hooi":
            core, factor_heads, factor_out, factor_in = _tucker_qkv_hooi(
                attention.in_proj_weight,
                attention.num_heads,
                head_rank,
                out_rank,
                in_rank,
                iterations=hooi_iterations,
            )
        elif tucker_method == "tensorly":
            core, factor_heads, factor_out, factor_in = _tucker_qkv_tensorly(
                attention.in_proj_weight,
                attention.num_heads,
                head_rank,
                out_rank,
                in_rank,
                iterations=hooi_iterations,
                tol=tensorly_tol,
            )
        else:
            raise ValueError(f"Unsupported Tucker method: {tucker_method}")
        with torch.no_grad():
            module.core.copy_(core)
            module.factor_heads.copy_(factor_heads)
            module.factor_out.copy_(factor_out)
            module.factor_in.copy_(factor_in)
            if attention.in_proj_bias is not None:
                module.qkv_bias.copy_(
                    attention.in_proj_bias.view(3, attention.num_heads, head_dim)
                )
        out_proj_rank = rank_from_ratio(attention.out_proj.weight, rank_ratio)
        module.out_proj = LowRankLinear.from_linear(attention.out_proj, out_proj_rank)
        return module

    def _project_qkv(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "bli,ir,qhdr,Hh,Dd->blqHD",
            inputs,
            self.factor_in,
            self.core,
            self.factor_heads,
            self.factor_out,
        ) + self.qkv_bias.view(1, 1, 3, self.num_heads, self.head_dim)

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
        if query is not key or query is not value:
            raise ValueError("TuckerSelfAttention only supports self-attention.")

        qkv = self._project_qkv(query)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
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


class HybridTuckerLightViT(LightViTBaseline):
    def __init__(self, rank_ratio: float = 0.1875, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < rank_ratio <= 1.0:
            raise ValueError("rank_ratio must be in the interval (0, 1].")
        self.rank_ratio = rank_ratio

    def _covariance_for(
        self,
        covariances: dict[str, torch.Tensor],
        name: str,
        expected_dim: int,
    ) -> torch.Tensor:
        covariance = covariances.get(name)
        if covariance is None:
            return torch.eye(expected_dim)
        if covariance.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Covariance for {name} must have shape "
                f"({expected_dim}, {expected_dim}), got {tuple(covariance.shape)}."
            )
        return covariance

    def apply_hybrid_tucker(
        self,
        covariances: dict[str, torch.Tensor] | None = None,
        rank_ratio: float | None = None,
        head_rank: int | None = None,
        tucker_method: str = "hosvd",
        hooi_iterations: int = 8,
        tensorly_tol: float = 1e-5,
    ) -> None:
        covariances = {} if covariances is None else covariances
        if rank_ratio is None:
            rank_ratio = self.rank_ratio
        if not 0.0 < rank_ratio <= 1.0:
            raise ValueError("rank_ratio must be in the interval (0, 1].")

        for layer_idx, block in enumerate(self.blocks.layers):
            block_name = f"blocks.layers.{layer_idx}"
            if isinstance(block.self_attn, nn.MultiheadAttention):
                block.self_attn = TuckerSelfAttention.from_multihead_attention(
                    block.self_attn,
                    rank_ratio,
                    head_rank=head_rank,
                    tucker_method=tucker_method,
                    hooi_iterations=hooi_iterations,
                    tensorly_tol=tensorly_tol,
                )

            if isinstance(block.linear1, nn.Linear):
                rank = rank_from_ratio(block.linear1.weight, rank_ratio)
                covariance = self._covariance_for(
                    covariances, f"{block_name}.linear1", block.linear1.in_features
                )
                block.linear1 = LowRankLinear.from_linear_activation_aware(
                    block.linear1, covariance, rank
                )

            if isinstance(block.linear2, nn.Linear):
                rank = rank_from_ratio(block.linear2.weight, rank_ratio)
                covariance = self._covariance_for(
                    covariances, f"{block_name}.linear2", block.linear2.in_features
                )
                block.linear2 = LowRankLinear.from_linear_activation_aware(
                    block.linear2, covariance, rank
                )

        if isinstance(self.head, nn.Linear):
            rank = rank_from_ratio(self.head.weight, rank_ratio)
            covariance = self._covariance_for(covariances, "head", self.head.in_features)
            self.head = LowRankLinear.from_linear_activation_aware(
                self.head, covariance, rank
            )
