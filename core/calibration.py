"""Forward-hook based activation capture for activation-aware compression."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn


class CovarianceCalibrator:
    def __init__(self) -> None:
        self.covariances: dict[str, torch.Tensor] = defaultdict(lambda: torch.empty(0))
        self.sample_counts: dict[str, int] = defaultdict(int)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, name: str):
        def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            features = inputs[0].detach().flatten(0, -2)
            covariance = features.transpose(0, 1) @ features
            if self.covariances[name].numel() == 0:
                self.covariances[name] = covariance
            else:
                self.covariances[name] = self.covariances[name] + covariance
            self.sample_counts[name] += features.shape[0]

        return capture

    def register(self, model: nn.Module, module_types: tuple[type[nn.Module], ...] = (nn.Linear,)) -> None:
        for name, module in model.named_modules():
            if isinstance(module, module_types):
                self._handles.append(module.register_forward_hook(self._hook(name)))

    def clear(self) -> None:
        self.covariances.clear()
        self.sample_counts.clear()

    def normalized_covariances(self) -> dict[str, torch.Tensor]:
        return {
            name: covariance / max(1, self.sample_counts[name])
            for name, covariance in self.covariances.items()
        }

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
