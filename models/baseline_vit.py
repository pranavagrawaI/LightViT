# models/baseline_vit.py
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_chans=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # A Conv2d stride equivalent to kernel_size perfectly extracts non-overlapping patches
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(
            1, 2
        )  # Reshape to [Batch, Sequence_Length, Embed_Dim]
        return x


class LightViTBaseline(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=2,
        in_chans=3,
        num_classes=100,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)

        # The classification token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )

        # The Transformer Core
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm is mandatory for deep ViT stability
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the Vision Transformer model.

        Args:
            x: Input tensor of shape (B, C, H, W) representing a batch of images.

        Returns:
            Output tensor of shape (B, num_classes) containing the class predictions
            for each image in the batch.

        Process:
            1. Extracts patches from the input images and embeds them
            2. Prepends a learnable [CLS] token to the sequence
            3. Adds positional embeddings to the patch embeddings
            4. Passes the sequence through transformer blocks
            5. Extracts the [CLS] token output and passes it through
               layer normalization and the classification head
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        # Extract only the [CLS] token's output for the final classification layer
        return self.head(self.norm(x[:, 0]))
