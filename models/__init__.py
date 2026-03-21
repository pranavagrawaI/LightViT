"""Model package for LightViT project scaffolding."""

from .baseline_vit import LightViTBaseline
from .compressed_vit import CompressedLightViT

__all__ = ["LightViTBaseline", "CompressedLightViT"]