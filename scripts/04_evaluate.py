"""Evaluate accuracy, speed, and compression metrics for the final model."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.metrics import count_parameters, measure_latency_ms
from models.compressed_vit import CompressedLightViT


def main() -> None:
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "final_qat_int8.pth"
    model = CompressedLightViT()

    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    sample = torch.randn(1, 3, 32, 32)
    params = count_parameters(model)
    latency_ms = measure_latency_ms(model, sample)
    print(f"Parameter count: {params}")
    print(f"Latency: {latency_ms:.3f} ms")


if __name__ == "__main__":
    main()