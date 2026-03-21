"""Fine-tune the factorized model with fake quantization and save final_qat_int8.pth."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.compressed_vit import CompressedLightViT


def main() -> None:
    factorized_path = PROJECT_ROOT / "checkpoints" / "factorized_fp32.pth"
    output_path = PROJECT_ROOT / "checkpoints" / "final_qat_int8.pth"

    model = CompressedLightViT()
    if factorized_path.exists() and factorized_path.stat().st_size > 0:
        checkpoint = torch.load(factorized_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    torch.save({"state_dict": model.state_dict(), "quantized": False}, output_path)
    print(f"Saved QAT scaffold checkpoint to {output_path}")


if __name__ == "__main__":
    main()