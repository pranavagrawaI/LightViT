"""Low-rank factorization helpers for compressed LightViT experiments."""

from __future__ import annotations

import torch


def factorize_linear_weight(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError("Expected a 2D linear weight tensor.")
    if rank <= 0:
        raise ValueError("rank must be positive.")

    left, singular_values, right_t = torch.linalg.svd(weight, full_matrices=False)
    left_rank = left[:, :rank] * singular_values[:rank]
    right_rank = right_t[:rank, :]
    return left_rank, right_rank


def activation_aware_rank(covariance: torch.Tensor, energy: float = 0.95) -> int:
    if covariance.ndim != 2:
        raise ValueError("Expected a 2D covariance matrix.")
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0).real.clamp_min(0)
    total_energy = eigenvalues.sum()
    if total_energy == 0:
        return 1
    retained = torch.cumsum(eigenvalues, dim=0) / total_energy
    return int((retained < energy).sum().item() + 1)