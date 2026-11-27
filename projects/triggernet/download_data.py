"""Download/load and preprocess the hls4ml 16-feature LHC jet tagging dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = DATA_DIR / "hls4ml_lhc_jets_hlf.npz"
PARQUET_PATH = DATA_DIR / "dataset_42468.pq"

# Canonical physics ordering used in hls4ml / FastML publications:
# g (gluon), q (light quark), w (W boson), z (Z boson), t (top quark)
CLASS_ORDER = ["g", "q", "w", "z", "t"]


def main() -> None:
    if CACHE_PATH.exists():
        print(f"Dataset already cached at {CACHE_PATH}")
        data = np.load(CACHE_PATH, allow_pickle=True)
        print("Shapes:", {k: data[k].shape for k in data.files if hasattr(data[k], "shape")})
        return

    if PARQUET_PATH.exists() and PARQUET_PATH.stat().st_size > 10_000_000:
        print(f"Loading OpenML Parquet dataset from {PARQUET_PATH} ...")
        df = pd.read_parquet(PARQUET_PATH)
        target_col = "class" if "class" in df.columns else df.columns[-1]
        feature_names = [c for c in df.columns if c != target_col]
    else:
        print("Fetching 'hls4ml_lhc_jets_hlf' from OpenML (830,000 LHC jets x 16 features)...")
        cache_dir = DATA_DIR / "openml_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        bunch = fetch_openml(
            "hls4ml_lhc_jets_hlf",
            version=1,
            as_frame=True,
            parser="auto",
            data_home=str(cache_dir),
        )
        df = bunch.frame
        feature_names = list(bunch.feature_names)
        target_col = bunch.target_names[0] if bunch.target_names else "class"

    X_raw = df[feature_names].to_numpy(dtype=np.float32)
    y_str = df[target_col].astype(str).to_numpy()

    # Map string labels ('g', 'q', 'w', 'z', 't') to canonical integer indices 0..4
    label_to_idx = {cls_name: idx for idx, cls_name in enumerate(CLASS_ORDER)}
    y_all = np.array([label_to_idx[label] for label in y_str], dtype=np.int64)

    # Stratified split: 60% train, 20% val, 20% test
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y_all, test_size=0.40, random_state=42, stratify=y_all
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Fit StandardScaler ONLY on training split (zero data leakage!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    np.savez_compressed(
        CACHE_PATH,
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=np.array(feature_names),
        class_names=np.array(CLASS_ORDER),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )
    print(f"Saved processed dataset to {CACHE_PATH}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    main()
