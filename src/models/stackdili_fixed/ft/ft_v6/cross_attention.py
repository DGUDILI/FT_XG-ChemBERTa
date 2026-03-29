import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """FP_16 (Query) × Chem_16 (Key/Value) → 16-dim fused representation.

    Q = FP projection  (1 token per sample)
    K = ChemBERTa projection
    V = ChemBERTa projection
    """

    def __init__(self, dim: int = 16, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0, f"dim({dim}) must be divisible by n_heads({n_heads})"
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

    def forward(self, fp_16: torch.Tensor, chem_16: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fp_16:   (B, 16) — FP projection
            chem_16: (B, 16) — ChemBERTa projection
        Returns:
            fused:   (B, 16)
        """
        q = fp_16.unsqueeze(1)    # (B, 1, 16)
        k = chem_16.unsqueeze(1)  # (B, 1, 16)
        v = chem_16.unsqueeze(1)  # (B, 1, 16)
        out, _ = self.attn(q, k, v)   # (B, 1, 16)
        return out.squeeze(1)          # (B, 16)
