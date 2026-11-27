"""CERN Next Generation Triggers `PQuantML` compression & Pareto sweep runner for Deep Sets."""
from __future__ import annotations

import copy
import os

# PQuantML selects its backend from KERAS_BACKEND at import time!
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import torch
import torch.nn as nn

import pquant
from pquant import (
    add_compression_layers,
    apply_final_compression,
    dst_config,
    get_ebops,
    get_layer_keep_ratio,
    get_model_losses,
    pdp_config,
    train_model,
)

from src.metrics import CLASS_NAMES, evaluate_jet_predictions


def build_pquant_config(
    total_bits: int,
    target_sparsity: float,
    pretrain_epochs: int = 0,
    prune_epochs: int = 10,
    finetune_epochs: int = 5,
    weight_int_bits: int = 1,
    data_int_bits: int = 2,
):
    """Build a verified CERN `PQuantML` config for fixed-point QAT + Progressive Differentiable Pruning (PDP).

    Bit-width breakdown in PQuantML:
      - Weight bits: `keep_negatives (1) + integer_bits (I) + fractional_bits (F) = total_bits`
      - Data/Activation bits: `keep_negatives (1) + integer_bits (I_d) + fractional_bits (F_d) = total_bits`
    """
    # Use pdp_config when pruning > 0; use dst_config with enable_pruning=False for 0% pruning
    # (Workaround for an off-by-one IndexError in pquant-ml 0.1.0 `_pdp_setup` when sparsity == 0.0)
    cfg = pdp_config() if target_sparsity > 0.0 else dst_config()

    # Quantization setup
    qp = cfg.quantization_parameters
    qp.enable_quantization = True
    qp.use_high_granularity_quantization = False
    qp.quantize_input = True
    qp.quantize_output = False
    qp.overflow_mode_parameters = "SAT"
    qp.overflow_mode_data = "SAT"
    qp.round_mode = "RND"

    w_int = min(weight_int_bits, max(0, total_bits - 2))
    w_frac = max(1, total_bits - 1 - w_int)
    d_int = min(data_int_bits, max(0, total_bits - 2))
    d_frac = max(1, total_bits - 1 - d_int)

    qp.default_weight_keep_negatives = 1.0
    qp.default_weight_integer_bits = float(w_int)
    qp.default_weight_fractional_bits = float(w_frac)
    qp.default_data_keep_negatives = 1.0
    qp.default_data_integer_bits = float(d_int)
    qp.default_data_fractional_bits = float(d_frac)

    # Pruning setup (Progressive Differentiable Pruning with exact target sparsity)
    # Note: In PQuantML PDP, sparsity ramps via min(1.0, epsilon * (epoch + 1)) * target_sparsity.
    # Setting epsilon = 1.0 / prune_epochs guarantees reaching exact target_sparsity cleanly!
    pp = cfg.pruning_parameters
    pp.enable_pruning = bool(target_sparsity > 0.0)
    if target_sparsity > 0.0:
        pp.sparsity = float(target_sparsity)
        pp.epsilon = 1.0 / max(1, prune_epochs)
        pp.temperature = 1e-5
        pp.structured_pruning = False

    # Multi-stage PQuantML training schedule
    tp = cfg.training_parameters
    tp.pretraining_epochs = int(pretrain_epochs)
    tp.rounds = 1
    tp.epochs = int(prune_epochs)
    tp.fine_tuning_epochs = int(finetune_epochs)
    tp.rewind = "never"

    return cfg


def run_pquant_compression_experiment(
    base_model: nn.Module,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    X_test_t: torch.Tensor,
    y_test_np: np.ndarray,
    total_bits: int,
    target_sparsity: float,
    prune_epochs: int = 10,
    finetune_epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 2e-3,
) -> tuple[nn.Module, dict]:
    """Compress a warm-started Deep Sets PyTorch model using CERN's `PQuantML` and evaluate."""
    device = X_train_t.device
    input_shape = (1,) + tuple(X_train_t.shape[1:])

    cfg = build_pquant_config(
        total_bits=total_bits,
        target_sparsity=target_sparsity,
        pretrain_epochs=0,
        prune_epochs=prune_epochs,
        finetune_epochs=finetune_epochs,
    )

    model = copy.deepcopy(base_model).to(device)
    model = add_compression_layers(model, cfg, input_shape).to(device)

    # Trigger lazy PQuantML quantizer initialization with one dummy batch
    with torch.no_grad():
        _ = model(X_train_t[:2])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_ep = max(1, prune_epochs + finetune_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_ep, eta_min=1e-4)

    def _train_epoch(m, epoch, **_kw):
        m.train()
        n = len(X_train_t)
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            sl = perm[start : start + batch_size]
            xb, yb = X_train_t[sl], y_train_t[sl]
            optimizer.zero_grad()
            logits = m(xb)
            loss = criterion(logits, yb)
            loss = get_model_losses(m, loss)
            loss.backward()
            optimizer.step()
        scheduler.step()

    val_hist = []

    def _valid_epoch(m, epoch, **_kw):
        m.eval()
        loss_sum, n_tot = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(X_val_t), batch_size * 2):
                xb = X_val_t[start : start + batch_size * 2]
                yb = y_val_t[start : start + batch_size * 2]
                loss_sum += criterion(m(xb), yb).item() * len(yb)
                n_tot += len(yb)
        val_hist.append(loss_sum / max(n_tot, 1))

    model = train_model(model, cfg, _train_epoch, _valid_epoch, input_shape=input_shape)
    apply_final_compression(model)

    # Evaluate on untouched test set
    model.eval()
    probs_list = []
    with torch.no_grad():
        for start in range(0, len(X_test_t), batch_size * 2):
            logits = model(X_test_t[start : start + batch_size * 2])
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)

    metrics = evaluate_jet_predictions(y_test_np, probs, CLASS_NAMES)
    keep_ratio = float(get_layer_keep_ratio(model))
    ebops = float(get_ebops(model))

    summary = {
        "tag": f"PQuant_{total_bits}b_p{int(round(target_sparsity * 100))}",
        "total_bits": int(total_bits),
        "target_sparsity": float(target_sparsity),
        "actual_sparsity": float(1.0 - keep_ratio),
        "keep_ratio": keep_ratio,
        "pquant_ebops": ebops,
        "accuracy": float(metrics["accuracy"]),
        "macro_auc": float(metrics["macro_auc"]),
        "mean_rej_50": float(metrics["mean_rej_50"]),
        "mean_rej_80": float(metrics["mean_rej_80"]),
        "per_class_auc": {c: float(metrics["per_class"][c]["auc"]) for c in CLASS_NAMES},
        "final_val_loss": float(val_hist[-1]) if val_hist else None,
    }
    return model, summary
