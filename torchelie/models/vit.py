import torch
import torch.nn as nn
from ..nn.transformer import ViTBlock


class ViTTrunk(nn.Module):
    """
    Vision Transformer (ViT) trunk that processes a sequence of patch embeddings with positional encoding
    and optional learnable registers, using a stack of ViTBlock layers.

    Args:
        seq_len (int): Length of the input sequence (number of patches).
        d_model (int): Dimension of the model.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        num_registers (int, optional): Number of learnable registers to prepend to the sequence. Default: 10.

    Forward Args:
        x (Tensor): Input tensor of shape [B, C, H/P, W/P], where P is the patch size.

    Returns:
        Tensor: Output tensor of shape [B, C, H/P, W/P].
    """

    def __init__(self, seq_len, d_model, num_layers, num_heads, num_registers=10):
        super().__init__()
        self.trunk = nn.ModuleList(
            [ViTBlock(d_model, num_heads) for _ in range(num_layers)]
        )
        self.pos_enc = nn.Parameter(torch.zeros(seq_len, d_model))
        self.registers = nn.Parameter(
            torch.randn(num_registers, d_model) / (d_model**0.5)
        )

    def forward(self, x):
        """
        Forward pass for the ViTTrunk.

        Args:
            x (Tensor): Input tensor of shape [B, C, H/P, W/P].

        Returns:
            Tensor: Output tensor of shape [B, C, H/P, W/P].
        """
        # x: [B,C,H/P,W/P]
        B, C, Hp, Wp = x.shape
        x = x.view(B, C, Hp * Wp).permute(0, 2, 1)
        x = x + self.pos_enc
        # x: [B, L, C]
        x = torch.cat([self.registers.unsqueeze(0).expand(B, -1, -1), x], dim=1)
        for block in self.trunk:
            x = block(x)

        x = x[:, len(self.registers) :, :]
        # x = F.gelu(x)
        x = x.permute(0, 2, 1).reshape(B, C, Hp, Wp)
        # x: [B,C,H/P,W/P]
        return x
