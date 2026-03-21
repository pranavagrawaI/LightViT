"""Strict measurement helpers for model size and latency."""

from __future__ import annotations

import time
import torch
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def measure_model_size_mb(model: nn.Module) -> float:
    """Calculates the exact physical memory footprint of the model weights in Megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size_mb = (param_size + buffer_size) / (1024**2)
    return size_mb


@torch.inference_mode()
def measure_latency_ms(
    model: nn.Module,
    sample: torch.Tensor,
    warmup_steps: int = 10,
    timed_steps: int = 50,
) -> float:
    model.eval()
    device = sample.device

    # Warmup Phase
    for _ in range(warmup_steps):
        model(sample)

    # Force synchronization before starting the clock
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()

    # Timed Phase
    for _ in range(timed_steps):
        model(sample)

    # Force synchronization before stopping the clock
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    end = time.perf_counter()
    return (end - start) * 1000.0 / timed_steps
