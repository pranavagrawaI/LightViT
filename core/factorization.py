"""Low-rank factorization helpers for compressed LightViT experiments."""

from __future__ import annotations

import torch


def factorize_linear_weight(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError("Expected a 2D linear weight tensor.")
    if rank <= 0:
        raise ValueError("rank must be positive.")
    rank = min(rank, min(weight.shape))

    left, singular_values, right_t = torch.linalg.svd(weight, full_matrices=False)
    left_rank = left[:, :rank] * singular_values[:rank]
    right_rank = right_t[:rank, :]
    return left_rank, right_rank


def factorize_linear_weight_act_svd(
    weight: torch.Tensor,
    covariance: torch.Tensor,
    rank: int,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError("Expected a 2D linear weight tensor.")
    if covariance.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Covariance shape must match the linear input dimension.")
    if rank <= 0:
        raise ValueError("rank must be positive.")

    rank = min(rank, min(weight.shape))
    covariance = covariance.to(device=weight.device, dtype=weight.dtype)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    diag_mean = covariance.diagonal().mean().clamp_min(eps)
    covariance = covariance / diag_mean

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(eps)
    covariance_root = eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.transpose(0, 1)
    covariance_inv_root = (
        eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.transpose(0, 1)
    )

    left, singular_values, right_t = torch.linalg.svd(
        weight @ covariance_root, full_matrices=False
    )
    left_rank = left[:, :rank] * singular_values[:rank]
    right_rank = right_t[:rank, :] @ covariance_inv_root
    return left_rank, right_rank


def rank_from_ratio(weight: torch.Tensor, rank_ratio: float) -> int:
    if weight.ndim != 2:
        raise ValueError("Expected a 2D linear weight tensor.")
    if not 0.0 < rank_ratio <= 1.0:
        raise ValueError("rank_ratio must be in the interval (0, 1].")
    return max(1, int(min(weight.shape) * rank_ratio))


def activation_aware_rank(covariance: torch.Tensor, energy: float = 0.95) -> int:
    if covariance.ndim != 2:
        raise ValueError("Expected a 2D covariance matrix.")
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0).real.clamp_min(0)
    total_energy = eigenvalues.sum()
    if total_energy == 0:
        return 1
    retained = torch.cumsum(eigenvalues, dim=0) / total_energy
    return int((retained < energy).sum().item() + 1)
