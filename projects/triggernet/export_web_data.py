"""Export real trained TriggerNet weights, LHC test jets, ROC curves, Pareto sweeps,
and generated hls4ml C++ firmware into lightweight JSON for the React interactive web app.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import JetMLP, QuantizedJetMLP
from src.metrics import CLASS_NAMES, CLASS_LABELS, evaluate_jet_predictions

OUT_DIR = PROJECT_ROOT / "web" / "public" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_seq_weights(seq: nn.Sequential) -> list[dict]:
    layers = []
    for mod in seq:
        if isinstance(mod, nn.Linear):
            w = mod.weight.detach().cpu().numpy()
            b = mod.bias.detach().cpu().numpy()
            layers.append({
                "in": int(mod.in_features),
                "out": int(mod.out_features),
                "w": np.round(w, 6).tolist(),
                "b": np.round(b, 6).tolist(),
                "nonzero": int((w != 0).sum()),
                "total": int(w.size),
            })
    return layers


def main() -> None:
    # 1. Load real test dataset
    data = np.load(PROJECT_ROOT / "data" / "hls4ml_lhc_jets_hlf.npz", allow_pickle=True)
    X_test_raw = data["X_test_raw"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = [str(f) for f in data["feature_names"]]

    # 2. Load real trained Level-1 models
    fp32 = JetMLP()
    fp32.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints" / "float32_baseline.pt", map_location="cpu", weights_only=True))
    fp32.eval()

    ckpt_8b = torch.load(PROJECT_ROOT / "checkpoints" / "qat_8b_p50.pt", map_location="cpu", weights_only=False)
    ckpt_6b = torch.load(PROJECT_ROOT / "checkpoints" / "qat_6b_p75.pt", map_location="cpu", weights_only=False)
    ckpt_4b = torch.load(PROJECT_ROOT / "checkpoints" / "qat_4b_p50.pt", map_location="cpu", weights_only=False)

    def _rebuild_seq(ckpt):
        seq = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 5),
        )
        seq.load_state_dict(ckpt["exported_state_dict"])
        seq.eval()
        return seq

    seq_8b = _rebuild_seq(ckpt_8b)
    seq_6b = _rebuild_seq(ckpt_6b)
    seq_4b = _rebuild_seq(ckpt_4b)

    # 3. Curate 25 real unseen LHC test jets (5 high-signal + interesting borderline cases per class)
    sample_jets = []
    mass_idx = feature_names.index("mass_mmdt")
    mult_idx = feature_names.index("multiplicity")
    c2_idx = feature_names.index("c2_b1_mmdt")
    d2_idx = feature_names.index("d2_b1_mmdt")

    with torch.no_grad():
        fp32_probs_all = torch.softmax(fp32(torch.from_numpy(X_test[:20000])), dim=1).numpy()

    for c_idx, cname in enumerate(CLASS_NAMES):
        cls_indices = np.where(y_test[:20000] == c_idx)[0]
        # Pick 3 clean textbook physics examples + 2 harder borderline examples per class
        conf = fp32_probs_all[cls_indices, c_idx]
        order = np.argsort(-conf)
        chosen = [
            cls_indices[order[5]],
            cls_indices[order[40]],
            cls_indices[order[150]],
            cls_indices[order[len(order) // 2]],
            cls_indices[order[int(len(order) * 0.72)]],
        ]
        for rank, idx in enumerate(chosen):
            sample_jets.append({
                "id": f"{cname.upper()}-{int(idx):05d}",
                "true_class_idx": int(c_idx),
                "true_class": cname,
                "true_label": CLASS_LABELS[cname],
                "difficulty": "Textbook" if rank < 3 else "Borderline",
                "mass_gev": round(float(X_test_raw[idx, mass_idx]), 2),
                "multiplicity": int(round(float(X_test_raw[idx, mult_idx]))),
                "c2_b1": round(float(X_test_raw[idx, c2_idx]), 4),
                "d2_b1": round(float(X_test_raw[idx, d2_idx]), 4),
                "x_scaled": np.round(X_test[idx], 5).tolist(),
                "x_raw": np.round(X_test_raw[idx], 4).tolist(),
            })

    # 4. Export compact ROC curves for FP32 vs QAT 6-bit+75% pruned (120 pts/curve)
    with torch.no_grad():
        qat_6b_full = QuantizedJetMLP(weight_bits=6, weight_int_bits=2, act_bits=6, act_int_bits=3)
        qat_6b_full.load_state_dict(ckpt_6b["state_dict"])
        qat_6b_full.eval()
        p_fp = torch.softmax(fp32(torch.from_numpy(X_test[:40000])), dim=1).numpy()
        p_q6 = torch.softmax(qat_6b_full(torch.from_numpy(X_test[:40000])), dim=1).numpy()

    m_fp = evaluate_jet_predictions(y_test[:40000], p_fp, CLASS_NAMES)
    m_q6 = evaluate_jet_predictions(y_test[:40000], p_q6, CLASS_NAMES)

    def _downsample_roc(fpr, tpr, n=100):
        grid = np.linspace(0.02, 1.0, n)
        idxs = np.searchsorted(tpr, grid)
        idxs = np.clip(idxs, 0, len(fpr) - 1)
        return [
            {"tpr": round(float(tpr[i]), 4), "fpr": round(float(max(fpr[i], 1e-4)), 5)}
            for i in idxs
        ]

    roc_payload = {}
    for c in CLASS_NAMES:
        roc_payload[c] = {
            "label": CLASS_LABELS[c],
            "fp32_auc": round(m_fp["per_class"][c]["auc"], 4),
            "q6_auc": round(m_q6["per_class"][c]["auc"], 4),
            "fp32_rej50": round(m_fp["per_class"][c]["rej_at_50"], 1),
            "q6_rej50": round(m_q6["per_class"][c]["rej_at_50"], 1),
            "fp32": _downsample_roc(m_fp["per_class"][c]["fpr"], m_fp["per_class"][c]["tpr"]),
            "q6": _downsample_roc(m_q6["per_class"][c]["fpr"], m_q6["per_class"][c]["tpr"]),
        }

    # 5. Export real generated hls4ml C++ firmware snippets
    fw_dir = PROJECT_ROOT / "hls_firmware" / "triggernet_6b_p75" / "firmware"
    defines_h = (fw_dir / "defines.h").read_text()
    project_cpp = (fw_dir / "triggernet_6b_p75.cpp").read_text()
    w2_raw = [float(v.strip()) for v in (fw_dir / "weights" / "w2.txt").read_text().strip().split(",") if v.strip()]

    # 6. Export models JSON (~110 KB) for real in-browser live inference
    models_payload = {
        "feature_names": feature_names,
        "class_names": CLASS_NAMES,
        "class_labels": CLASS_LABELS,
        "models": {
            "fp32": {
                "name": "Float32 Baseline",
                "bits": 32,
                "w_int": 8,
                "a_bits": 32,
                "a_int": 8,
                "sparsity_pct": 0.0,
                "bops": 4358144,
                "bop_reduction_pct": 0.0,
                "test_acc": 76.71,
                "macro_auc": 0.9431,
                "rej_50": 284.1,
                "layers": extract_seq_weights(fp32.net),
            },
            "qat_8b_p50": {
                "name": "QAT 8-bit + 50% Pruned",
                "bits": 8,
                "w_int": 2,
                "a_bits": 8,
                "a_int": 3,
                "sparsity_pct": 50.0,
                "bops": 136192,
                "bop_reduction_pct": 96.88,
                "test_acc": 76.72,
                "macro_auc": 0.9427,
                "rej_50": 279.8,
                "layers": extract_seq_weights(seq_8b),
            },
            "qat_6b_p75": {
                "name": "QAT 6-bit + 75% Pruned",
                "bits": 6,
                "w_int": 2,
                "a_bits": 6,
                "a_int": 3,
                "sparsity_pct": 75.0,
                "bops": 38304,
                "bop_reduction_pct": 99.12,
                "test_acc": 75.93,
                "macro_auc": 0.9381,
                "rej_50": 202.7,
                "layers": extract_seq_weights(seq_6b),
            },
            "qat_4b_p50": {
                "name": "QAT 4-bit + 50% Pruned",
                "bits": 4,
                "w_int": 2,
                "a_bits": 4,
                "a_int": 3,
                "sparsity_pct": 52.3,
                "bops": 32496,
                "bop_reduction_pct": 99.25,
                "test_acc": 74.79,
                "macro_auc": 0.9314,
                "rej_50": 145.6,
                "layers": extract_seq_weights(seq_4b),
            },
        },
        "sample_jets": sample_jets,
    }
    (OUT_DIR / "inference_models.json").write_text(json.dumps(models_payload))

    # 7. Export unified metrics + sweeps + firmware bundle
    l1_sweep = json.loads((PROJECT_ROOT / "results" / "compression_sweep.json").read_text())
    hls_ver = json.loads((PROJECT_ROOT / "results" / "hls_verification.json").read_text())
    l2_real_path = PROJECT_ROOT / "results" / "level2_pquant_sweep_real.json"
    l2_path = l2_real_path if l2_real_path.exists() else (PROJECT_ROOT / "results" / "level2_pquant_sweep.json")
    l2_sweep = json.loads(l2_path.read_text())

    dashboard_payload = {
        "ptq_vs_qat": [
            {"bits": 8, "label": "ap_fixed<8,2>", "ptq_acc": 76.70, "qat_acc": 76.73, "ptq_auc": 0.9428, "qat_auc": 0.9432},
            {"bits": 6, "label": "ap_fixed<6,2>", "ptq_acc": 75.55, "qat_acc": 76.44, "ptq_auc": 0.9384, "qat_auc": 0.9413},
            {"bits": 4, "label": "ap_fixed<4,2>", "ptq_acc": 57.52, "qat_acc": 74.47, "ptq_auc": 0.8784, "qat_auc": 0.9301},
        ],
        "level1_sweep": l1_sweep,
        "level2_sweep": l2_sweep,
        "hls_verification": hls_ver,
        "roc_curves": roc_payload,
        "firmware": {
            "defines_h": defines_h,
            "project_cpp": project_cpp,
            "w2_sample": w2_raw[:128],
            "w2_zero_pct": round(100.0 * sum(v == 0.0 for v in w2_raw) / len(w2_raw), 1),
            "w2_total": len(w2_raw),
        },
    }
    (OUT_DIR / "dashboard_data.json").write_text(json.dumps(dashboard_payload))

    print("Exported web JSON bundles:")
    for p in sorted(OUT_DIR.glob("*.json")):
        print(f"  - {p.name}: {p.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
