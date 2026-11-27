"""Permutation-invariant particle-set models for TriggerNet Level 2 (constituent jets)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DeepSetsJetTagger(nn.Module):
    """Permutation-invariant Deep Sets jet classifier (Zaheer et al.; Odagiu et al. 2024).

    Physics & Math Intuition:
        A jet is an UNORDERED set of N constituent particles {x_1, x_2, ..., x_N},
        where each particle has kinematic features (e.g., pT_rel, delta_eta, delta_phi).
        Reordering the particles inside a jet does NOT change whether it came from a
        gluon, quark, W, Z, or top quark.

        Deep Sets guarantees exact permutation invariance via:
            f({x_1, ..., x_N}) = rho( Pool_i( phi(x_i) ) )

        1. `phi` (shared per-particle MLP): processes every particle independently
           using the EXACT same weights: (B, N, F) -> (B, N, hidden_dim).
        2. `Pool` (masked mean or sum): combines all particles symmetrically across
           the particle axis N -> (B, hidden_dim). Because addition/averaging is
           commutative (a + b = b + a), shuffling particle rows cannot change the output!
        3. `rho` (jet classifier MLP): maps the pooled jet embedding -> 5 class logits.

    Why this architecture wins for CERN FPGAs:
        - Tiny parameter count (~3.5k params at hidden=32, matching Odagiu et al. 2024).
        - Every layer is a pure `nn.Linear` + `nn.ReLU`, which CERN's `PQuantML` and
          `hls4ml` compress and unroll directly onto FPGA silicon!
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_dim: int = 32,
        num_classes: int = 5,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if aggregation not in ("mean", "sum"):
            raise ValueError(f"aggregation must be 'mean' or 'sum', got {aggregation}")
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.aggregation = aggregation

        # phi: per-particle feature extractor (applied identically to all N particles)
        self.phi_fc1 = nn.Linear(in_features, hidden_dim)
        self.phi_act1 = nn.ReLU()
        self.phi_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.phi_act2 = nn.ReLU()
        self.phi_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.phi_act3 = nn.ReLU()

        # rho: jet-level classifier after permutation-invariant pooling
        self.rho_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.rho_act1 = nn.ReLU()
        self.rho_fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on constituent tensor `x` of shape `(B, N, in_features)`."""
        # Zero-padded particles (all features == 0) are masked out cleanly so
        # padding never corrupts the pooled jet representation.
        mask = (x.abs().sum(dim=-1, keepdim=True) > 1e-6).to(x.dtype)

        h = self.phi_act1(self.phi_fc1(x))
        h = self.phi_act2(self.phi_fc2(h))
        h = self.phi_act3(self.phi_fc3(h))
        h = h * mask

        if self.aggregation == "mean":
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = h.sum(dim=1) / denom
        else:
            pooled = h.sum(dim=1)

        out = self.rho_act1(self.rho_fc1(pooled))
        return self.rho_fc2(out)


class ParticleSetTransformer(nn.Module):
    """Lightweight order-free Set Transformer (1 self-attention block, no positional encoding).

    Why NO positional encoding?
        Standard NLP transformers add positional embeddings because word order matters.
        In a particle jet, constituent order is arbitrary! Dropping positional encodings
        makes Multi-Head Self-Attention permutation-equivariant: every particle attends
        to pairwise angular/momentum relationships with all other particles (e.g., spotting
        2-prong W/Z or 3-prong top sub-clusters), followed by symmetric mean pooling.
    """

    def __init__(
        self,
        in_features: int = 3,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_features, embed_dim)
        self.input_act = nn.ReLU()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn_fc1 = nn.Linear(embed_dim, embed_dim * 2)
        self.ffn_act = nn.ReLU()
        self.ffn_fc2 = nn.Linear(embed_dim * 2, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.head_fc1 = nn.Linear(embed_dim, embed_dim)
        self.head_act = nn.ReLU()
        self.head_fc2 = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        valid_mask = (x.abs().sum(dim=-1) > 1e-6)  # (B, N) True where real particle
        # Avoid all-True padding mask edge case
        pad_mask = ~valid_mask
        all_pad = pad_mask.all(dim=1)
        if all_pad.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_pad, 0] = False

        h = self.input_act(self.input_proj(x))
        attn_out, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn_fc2(self.ffn_act(self.ffn_fc1(h))))

        m = valid_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head_fc2(self.head_act(self.head_fc1(pooled)))
