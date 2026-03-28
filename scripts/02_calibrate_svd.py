"""Calibrate activations, apply low-rank factorization, and save factorized_fp32.pth."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.calibration import CovarianceCalibrator
from models.compressed_vit import CompressedLightViT


def main() -> None:
    baseline_path = PROJECT_ROOT / "checkpoints" / "baseline_fp32.pth"
    output_path = PROJECT_ROOT / "checkpoints" / "factorized_fp32.pth"

    model = CompressedLightViT()
    if baseline_path.exists() and baseline_path.stat().st_size > 0:
        checkpoint = torch.load(baseline_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    calibrator = CovarianceCalibrator()
    calibrator.register(model)
    sample = torch.randn(1, 3, 32, 32)
    model(sample)
    calibrator.remove()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "covariance_keys": sorted(calibrator.covariances.keys()),
        },
        output_path,
    )
    print(f"Saved factorized scaffold checkpoint to {output_path}")


if __name__ == "__main__":
    main()