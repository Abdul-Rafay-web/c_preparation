"""PyTorch Jet Tagging MLP, Fixed-Point Quantization-Aware Training (QAT), and Pruning."""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class JetMLP(nn.Module):
    """Canonical hls4ml 3-hidden-layer jet tagging MLP: 16 -> 64 -> 32 -> 32 -> 5.

    Why this exact size?
    Every neuron multiplication consumes FPGA DSP slices / LUTs. Keeping hidden
    widths at 64 -> 32 -> 32 (~4,389 parameters) lets the whole network run
    fully unrolled in nanoseconds on a Level-1 trigger FPGA while capturing
    non-linear jet substructure correlations.
    """

    def __init__(self, in_features: int = 16, num_classes: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fixed_point_quantize_ste(
    x: torch.Tensor,
    total_bits: int,
    int_bits: int,
    signed: bool = True,
) -> torch.Tensor:
    """Hardware-exact ap_fixed<total_bits, int_bits> quantizer with Straight-Through Estimator.

    In FPGA HLS (Vivado/Vitis `ap_fixed<W, I>`):
      - `total_bits` (W) = total binary bits per number.
      - `int_bits`   (I) = integer bits INCLUDING the sign bit (if signed).
      - `frac_bits`  (F) = W - I = fractional bits (step size = 2^-F).

    Range for signed `ap_fixed<W, I>`:
      [-2^(I-1),  2^(I-1) - 2^(-F)]
    Range for unsigned `ap_ufixed<W, I>` (ideal after ReLU since ReLU >= 0!):
      [0,         2^I - 2^(-F)]

    Why Straight-Through Estimator (STE)?
    `torch.round()` has derivative 0 almost everywhere, which would kill backprop.
    STE runs the rounded fixed-point value in the forward pass, while pretending
    the derivative is 1 (inside the clamp range) in the backward pass:
        y = x + (q(x) - x).detach()
    """
    frac_bits = total_bits - int_bits
    scale = float(2 ** frac_bits)

    if signed:
        min_val = -float(2 ** (int_bits - 1))
        max_val = float(2 ** (int_bits - 1)) - (1.0 / scale)
    else:
        min_val = 0.0
        max_val = float(2 ** int_bits) - (1.0 / scale)

    x_clamped = torch.clamp(x, min_val, max_val)
    x_q = torch.round(x_clamped * scale) / scale
    # Forward pass uses exact quantized x_q; backward pass flows gradients through x_clamped
    return x_clamped + (x_q - x_clamped).detach()


class QFixedLinear(nn.Linear):
    """Linear layer with fixed-point ap_fixed<W, I> weight/bias QAT and optional pruning mask."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_bits: int = 8,
        weight_int_bits: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.weight_bits = weight_bits
        self.weight_int_bits = weight_int_bits
        self.register_buffer("weight_mask", torch.ones_like(self.weight))

    def get_quantized_weights(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        masked_w = self.weight * self.weight_mask
        q_w = fixed_point_quantize_ste(
            masked_w, total_bits=self.weight_bits, int_bits=self.weight_int_bits, signed=True
        )
        # Re-apply mask so pruned zeros stay strictly 0.0
        q_w = q_w * self.weight_mask
        q_b = None
        if self.bias is not None:
            q_b = fixed_point_quantize_ste(
                self.bias, total_bits=self.weight_bits, int_bits=self.weight_int_bits, signed=True
            )
        return q_w, q_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_w, q_b = self.get_quantized_weights()
        return F.linear(x, q_w, q_b)


class QFixedReLU(nn.Module):
    """ReLU followed by unsigned fixed-point ap_ufixed<W, I> quantization."""

    def __init__(self, act_bits: int = 8, act_int_bits: int = 3) -> None:
        super().__init__()
        self.act_bits = act_bits
        self.act_int_bits = act_int_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        return fixed_point_quantize_ste(
            x, total_bits=self.act_bits, int_bits=self.act_int_bits, signed=False
        )


class QuantizedJetMLP(nn.Module):
    """Quantization-Aware + Prunable version of JetMLP matching FPGA ap_fixed arithmetic."""

    def __init__(
        self,
        in_features: int = 16,
        num_classes: int = 5,
        weight_bits: int = 8,
        weight_int_bits: int = 2,
        act_bits: int = 8,
        act_int_bits: int = 3,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.weight_int_bits = weight_int_bits
        self.act_bits = act_bits
        self.act_int_bits = act_int_bits

        self.net = nn.Sequential(
            QFixedLinear(in_features, 64, weight_bits, weight_int_bits),
            QFixedReLU(act_bits, act_int_bits),
            QFixedLinear(64, 32, weight_bits, weight_int_bits),
            QFixedReLU(act_bits, act_int_bits),
            QFixedLinear(32, 32, weight_bits, weight_int_bits),
            QFixedReLU(act_bits, act_int_bits),
            QFixedLinear(32, num_classes, weight_bits, weight_int_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def apply_global_magnitude_pruning(self, target_sparsity: float) -> float:
        """Zero out the smallest `target_sparsity` fraction of weights across all Linear layers."""
        if target_sparsity <= 0.0:
            for m in self.modules():
                if isinstance(m, QFixedLinear):
                    m.weight_mask.fill_(1.0)
            return 0.0

        all_abs_weights = torch.cat(
            [m.weight.detach().abs().flatten() for m in self.modules() if isinstance(m, QFixedLinear)]
        )
        k = int(round(target_sparsity * all_abs_weights.numel()))
        if k <= 0:
            return 0.0
        threshold = torch.kthvalue(all_abs_weights, k).values.item()

        total = 0
        zeros = 0
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, QFixedLinear):
                    mask = (m.weight.abs() > threshold).to(m.weight.dtype)
                    m.weight_mask.copy_(mask)
                    m.weight.mul_(mask)
                    total += mask.numel()
                    zeros += int((mask == 0).sum().item())
        return zeros / max(total, 1)

    def export_to_standard_sequential(self) -> nn.Sequential:
        """Bake quantized + pruned weights into a clean nn.Sequential for hls4ml conversion."""
        layers: list[nn.Module] = []
        with torch.no_grad():
            for module in self.net:
                if isinstance(module, QFixedLinear):
                    lin = nn.Linear(module.in_features, module.out_features, bias=(module.bias is not None))
                    q_w, q_b = module.get_quantized_weights()
                    lin.weight.copy_(q_w)
                    if q_b is not None and lin.bias is not None:
                        lin.bias.copy_(q_b)
                    layers.append(lin)
                elif isinstance(module, QFixedReLU):
                    layers.append(nn.ReLU())
        return nn.Sequential(*layers)


def load_float_weights_into_qat(float_model: JetMLP, qat_model: QuantizedJetMLP) -> None:
    """Warm-start QAT model from a trained float32 baseline for faster, cleaner convergence."""
    float_linears = [m for m in float_model.modules() if isinstance(m, nn.Linear)]
    qat_linears = [m for m in qat_model.modules() if isinstance(m, QFixedLinear)]
    with torch.no_grad():
        for src, dst in zip(float_linears, qat_linears):
            dst.weight.copy_(src.weight)
            if src.bias is not None and dst.bias is not None:
                dst.bias.copy_(src.bias)
