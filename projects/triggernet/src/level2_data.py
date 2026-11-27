"""Constituent-level LHC jet dataset loader (Zenodo 3602260 + fast local dry-run mode).

Zenodo 3602260 (`hls4ml_LHCjet_150p`):
    - Total size: ~3.87 GB across `hls4ml_LHCjet_150p_train.tar.gz` (2.73 GB) and
      `hls4ml_LHCjet_150p_val.tar.gz` (1.14 GB).
    - Contains HDF5 shards (`jetImage_*.h5`), each holding:
        * `jetConstituentList`: shape `(N_jets, 150, 16)` (up to 150 particles per jet,
          sorted by descending pT, zero-padded when a jet has fewer than 150 particles).
        * `jets`: one-hot class flags at columns `[-6:-1]` corresponding to
          `[j_g, j_q, j_w, j_z, j_t]` (indices 0..4).
    - The 3 canonical trigger constituent features (Odagiu et al., MLST 2024):
        * Feature 5 (`j1_ptrel`): particle pT divided by jet pT (relative energy fraction)
        * Feature 8 (`j1_etarel`): particle delta-eta relative to jet axis
        * Feature 11 (`j1_phirel`): particle delta-phi relative to jet axis
      (We also verify feature names dynamically from the HDF5 `particleFeatureNames` key!)
"""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
import numpy as np

ZENODO_BASE = "https://zenodo.org/records/3602260/files"
ZENODO_FILES = {
    "train": "hls4ml_LHCjet_150p_train.tar.gz",
    "val": "hls4ml_LHCjet_150p_val.tar.gz",
}

CLASS_NAMES = ["g", "q", "w", "z", "t"]
TARGET_COL_NAMES = ["j_g", "j_q", "j_w", "j_z", "j_t"]
CANONICAL_3_FEATURES = ["j1_ptrel", "j1_etarel", "j1_phirel"]

# Published reference table from Odagiu et al. (2024),
# "Ultrafast jet classification at the HL-LHC", MLST 5(3) 035017.
PUBLISHED_ODAGIU_2024 = [
    {"paper_model": "Odagiu2024 Deep Sets (N=8, FP32)", "n_const": 8, "params": 3461, "acc": 0.640, "macro_auc": 0.884},
    {"paper_model": "Odagiu2024 Deep Sets (N=16, FP32)", "n_const": 16, "params": 3461, "acc": 0.695, "macro_auc": 0.900},
    {"paper_model": "Odagiu2024 Deep Sets (N=32, FP32)", "n_const": 32, "params": 3461, "acc": 0.735, "macro_auc": 0.918},
]


def download_zenodo_3602260(data_dir: Path) -> tuple[Path, Path]:
    """Download and extract Zenodo 3602260 train and test/val tarballs (best run on GPU cloud)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted_dirs = {}
    for split, fname in ZENODO_FILES.items():
        tar_path = data_dir / fname
        out_dir = data_dir / split
        if not out_dir.exists() or not list(out_dir.rglob("*.h5")):
            if not tar_path.exists():
                url = f"{ZENODO_BASE}/{fname}?download=1"
                print(f"Downloading {fname} from Zenodo 3602260 ...")
                urllib.request.urlretrieve(url, tar_path)
            print(f"Extracting {tar_path.name} -> {out_dir} ...")
            out_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(out_dir)
        extracted_dirs[split] = out_dir
    return extracted_dirs["train"], extracted_dirs["val"]


def _decode_str_list(arr) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]


def load_h5_folder(
    folder: Path,
    n_constituents: int = 32,
    feature_names: list[str] | None = None,
    max_files: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read `hls4ml_LHCjet_150p` HDF5 shards into `(X, y, selected_feature_names)`."""
    import h5py

    h5_files = sorted(folder.rglob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {folder}")
    if max_files is not None:
        h5_files = h5_files[:max_files]

    selected_feats = feature_names or CANONICAL_3_FEATURES
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    feat_idx = None
    label_idx = None

    for fp in h5_files:
        with h5py.File(fp, "r") as f:
            if feat_idx is None:
                p_names = _decode_str_list(f["particleFeatureNames"][:])
                j_names = _decode_str_list(f["jetFeatureNames"][:])
                feat_idx = [p_names.index(name) for name in selected_feats]
                label_idx = [j_names.index(name) for name in TARGET_COL_NAMES]

            const = f["jetConstituentList"][:, :n_constituents, :]
            const = const[:, :, feat_idx].astype(np.float32)
            jets = f["jets"][:]
            y_onehot = jets[:, label_idx]
            y_int = np.argmax(y_onehot, axis=1).astype(np.int64)

            x_chunks.append(const)
            y_chunks.append(y_int)

    X = np.concatenate(x_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    return X, y, selected_feats


def standardize_constituents(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize non-padded particles using ONLY training-split statistics (zero leakage).

    Zero-padded particle slots (where all features == 0) remain strictly 0.0 so the
    Deep Sets padding mask works identically before and after scaling.
    """
    train_mask = np.abs(X_train).sum(axis=-1) > 1e-6
    valid_pixels = X_train[train_mask]  # (num_valid_particles, F)
    mean = valid_pixels.mean(axis=0).astype(np.float32)
    std = np.maximum(valid_pixels.std(axis=0), 1e-6).astype(np.float32)

    def _apply(arr: np.ndarray) -> np.ndarray:
        mask = (np.abs(arr).sum(axis=-1, keepdims=True) > 1e-6)
        scaled = (arr - mean) / std
        return np.where(mask, scaled, 0.0).astype(np.float32)

    return _apply(X_train), _apply(X_val), _apply(X_test), mean, std


def generate_synthetic_constituent_jets(
    n_jets: int = 12_000,
    n_constituents: int = 32,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fast physics-inspired synthetic jet constituents for LOCAL pipeline dry-runs ONLY.

    Simulates realistic multi-prong angular patterns in (pT_rel, delta_eta, delta_phi):
      - Gluon (g): broad single cluster, high multiplicity
      - Quark (q): tight single cluster, lower multiplicity
      - W Boson (w): 2-prong sub-clusters separated by delta_R ~ 0.16
      - Z Boson (z): 2-prong sub-clusters separated by delta_R ~ 0.19
      - Top Quark (t): 3-prong sub-clusters separated by delta_R ~ 0.30
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_jets, n_constituents, 3), dtype=np.float32)
    y = rng.integers(0, 5, size=n_jets, dtype=np.int64)

    for i in range(n_jets):
        cls = int(y[i])
        if cls == 0:  # gluon: broad 1-prong, high multiplicity
            m = int(rng.integers(max(8, n_constituents // 2), n_constituents + 1))
            centers = [(0.0, 0.0)]
            spread = 0.13
        elif cls == 1:  # quark: narrow 1-prong, lower multiplicity
            m = int(rng.integers(6, max(10, int(n_constituents * 0.65))))
            centers = [(0.0, 0.0)]
            spread = 0.065
        elif cls == 2:  # W boson: 2-prong
            m = int(rng.integers(8, max(12, int(n_constituents * 0.8))))
            ang = rng.uniform(0, 2 * np.pi)
            d = rng.normal(0.085, 0.015)
            centers = [(d * np.cos(ang), d * np.sin(ang)), (-d * np.cos(ang), -d * np.sin(ang))]
            spread = 0.045
        elif cls == 3:  # Z boson: slightly wider 2-prong
            m = int(rng.integers(8, max(12, int(n_constituents * 0.8))))
            ang = rng.uniform(0, 2 * np.pi)
            d = rng.normal(0.105, 0.015)
            centers = [(d * np.cos(ang), d * np.sin(ang)), (-d * np.cos(ang), -d * np.sin(ang))]
            spread = 0.045
        else:  # top quark: 3-prong
            m = int(rng.integers(max(12, n_constituents // 2), n_constituents + 1))
            ang0 = rng.uniform(0, 2 * np.pi)
            d = rng.normal(0.155, 0.025)
            centers = [
                (d * np.cos(ang0 + k * 2 * np.pi / 3), d * np.sin(ang0 + k * 2 * np.pi / 3))
                for k in range(3)
            ]
            spread = 0.055

        m = min(m, n_constituents)
        c_ids = rng.integers(0, len(centers), size=m)
        etas = np.array([centers[c][0] for c in c_ids]) + rng.normal(0, spread, size=m)
        phis = np.array([centers[c][1] for c in c_ids]) + rng.normal(0, spread, size=m)
        pts = rng.gamma(shape=1.4, scale=1.0, size=m)
        pts = pts / pts.sum()
        order = np.argsort(-pts)
        X[i, :m, 0] = pts[order]
        X[i, :m, 1] = etas[order]
        X[i, :m, 2] = phis[order]

    return X, y, CANONICAL_3_FEATURES
