"""Physics-aware evaluation metrics for LHC jet trigger classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

CLASS_NAMES = ["g", "q", "w", "z", "t"]
CLASS_LABELS = {
    "g": "Gluon (g)",
    "q": "Light Quark (q)",
    "w": "W Boson (W)",
    "z": "Z Boson (Z)",
    "t": "Top Quark (t)",
}


def background_rejection_at_eff(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    target_eff: float = 0.50,
) -> tuple[float, float]:
    """Compute background rejection (1 / FPR) at a fixed signal efficiency (TPR).

    In LHC triggers, physicists ask:
        "If we keep `target_eff` (e.g. 50% or 80%) of true signal jets,
         by what factor do we suppress the overwhelming background?"

    Returns:
        (rejection_factor, fpr_at_target) where rejection_factor = 1 / max(fpr, 1e-6).
    """
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    idx = int(np.searchsorted(tpr, target_eff, side="left"))
    idx = min(idx, len(fpr) - 1)
    fpr_val = float(fpr[idx])
    rejection = float(1.0 / max(fpr_val, 1e-6))
    return rejection, fpr_val


def evaluate_jet_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Compute standard CERN trigger metrics: accuracy, per-class ROC/AUC, and 1/FPR."""
    names = class_names or CLASS_NAMES
    preds = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, preds))

    per_class = {}
    aucs = []
    rej_50_list = []
    rej_80_list = []

    for i, name in enumerate(names):
        y_bin = (y_true == i).astype(np.int32)
        score = probs[:, i]
        fpr, tpr, _ = roc_curve(y_bin, score)
        auc_val = float(roc_auc_score(y_bin, score))
        rej_50, fpr_50 = background_rejection_at_eff(y_bin, score, target_eff=0.50)
        rej_80, fpr_80 = background_rejection_at_eff(y_bin, score, target_eff=0.80)

        aucs.append(auc_val)
        rej_50_list.append(rej_50)
        rej_80_list.append(rej_80)
        per_class[name] = {
            "auc": auc_val,
            "rej_at_50": rej_50,
            "fpr_at_50": fpr_50,
            "rej_at_80": rej_80,
            "fpr_at_80": fpr_80,
            "fpr": fpr,
            "tpr": tpr,
        }

    return {
        "accuracy": acc,
        "macro_auc": float(np.mean(aucs)),
        "mean_rej_50": float(np.mean(rej_50_list)),
        "mean_rej_80": float(np.mean(rej_80_list)),
        "per_class": per_class,
    }


def compute_model_hardware_stats(model, weight_bits: int = 32, act_bits: int = 32) -> dict:
    """Estimate FPGA-relevant size and Bit-Operations (BOPs) for a Linear MLP.

    On an FPGA, a weight that is exactly 0 requires ZERO DSP multipliers after
    constant folding in HLS. Effective BOPs therefore scales with:
        non_zero_MACs * weight_bits * act_bits.
    """
    import torch.nn as nn

    total_weights = 0
    nonzero_weights = 0
    total_biases = 0
    total_macs = 0
    nonzero_macs = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.detach()
            n_w = int(w.numel())
            nz_w = int((w != 0).sum().item())
            total_weights += n_w
            nonzero_weights += nz_w
            total_macs += n_w
            nonzero_macs += nz_w
            if module.bias is not None:
                total_biases += int(module.bias.numel())

    sparsity = 1.0 - (nonzero_weights / max(total_weights, 1))
    bops = nonzero_macs * weight_bits * act_bits
    # Equivalent storage in KB for non-zero weights + biases
    weight_kb = (nonzero_weights * weight_bits + total_biases * weight_bits) / (8.0 * 1024.0)

    return {
        "total_weights": total_weights,
        "nonzero_weights": nonzero_weights,
        "total_params": total_weights + total_biases,
        "sparsity": float(sparsity),
        "weight_bits": int(weight_bits),
        "act_bits": int(act_bits),
        "effective_bops": int(bops),
        "weight_kb": float(weight_kb),
    }
