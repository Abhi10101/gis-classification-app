# ============================================================
# core/feature_engineering.py
# ============================================================

import json
from pathlib import Path
import numpy as np


# ------------------------------------------------------------
# OPTIONAL IMPORTS (NO SILENT LOGIC CHANGE)
# ------------------------------------------------------------
try:
    from logger import log_event
except Exception:
    def log_event(msg: str):
        pass

try:
    from validators import validate_feature_matrix
except Exception:
    validate_feature_matrix = None


def prepare_features(
    X_path: str,
    y_path: str,
    out_dir: str,
    normalize: bool = False
):
    """
    DO NOT CHANGE SIGNATURE
    """

    # --------------------------------------------------------
    # OUTPUT DIRECTORY (OWNED BY APP)
    # --------------------------------------------------------
    out_dir = Path(out_dir)

    if not out_dir.exists():
        raise FileNotFoundError(
            f"Output directory does not exist (must be created by app.py): {out_dir}"
        )

    if not out_dir.is_dir():
        raise NotADirectoryError(f"out_dir is not a directory: {out_dir}")

    # --------------------------------------------------------
    # INPUT PATH VALIDATION
    # --------------------------------------------------------
    X_path = Path(X_path)
    y_path = Path(y_path)

    if not X_path.exists():
        raise FileNotFoundError(f"X.npy not found: {X_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"y.npy not found: {y_path}")

    log_event("Preparing feature matrix")

    # --------------------------------------------------------
    # LOAD ARRAYS (NO CAST YET)
    # --------------------------------------------------------
    X = np.load(X_path)
    y = np.load(y_path)

    if validate_feature_matrix is not None:
        validate_feature_matrix(X, y)

    # --------------------------------------------------------
    # SHAPE VALIDATION
    # --------------------------------------------------------
    if X.ndim != 2:
        raise ValueError("X must be 2D [n_samples, n_features]")

    if y.ndim != 1:
        raise ValueError("y must be 1D")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y length mismatch")

    n_samples, n_features = X.shape

    if n_samples == 0 or n_features == 0:
        raise ValueError("Empty feature matrix")

    # --------------------------------------------------------
    # NDVI COLUMN VALIDATION (LAST COLUMN)
    # --------------------------------------------------------
    ndvi_col = n_features - 1
    ndvi_vals = X[:, ndvi_col]

    if not np.isfinite(ndvi_vals).all():
        raise ValueError("NDVI column contains NaN or Inf values")

    # --------------------------------------------------------
    # REMOVE INVALID ROWS (EXPLICIT, NO SILENT DROP)
    # --------------------------------------------------------
    valid_mask = np.isfinite(X).all(axis=1)

    removed = int((~valid_mask).sum())

    if removed == n_samples:
        raise RuntimeError("All samples invalid after finite check")

    X = X[valid_mask]
    y = y[valid_mask]

    # --------------------------------------------------------
    # NORMALIZATION (OPTIONAL, DETERMINISTIC)
    # --------------------------------------------------------
    scaler = None

    if normalize:
        # cast ONCE before stats
        X = X.astype(np.float32, copy=False)

        mean = X.mean(axis=0)
        std = X.std(axis=0)

        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Non-finite mean/std during normalization")

        if (std == 0).any():
            raise ValueError("Zero-variance feature detected")

        X = (X - mean) / std

        scaler = {
            "type": "standard",
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    else:
        X = X.astype(np.float32, copy=False)

    y = y.astype(np.int64, copy=False)

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------
    report = {
        "initial_samples": int(n_samples),
        "final_samples": int(len(X)),
        "removed_invalid_rows": removed,
        "n_features": int(n_features),
        "ndvi_column_index": int(ndvi_col),
        "normalized": bool(normalize),
        "scaler": scaler["type"] if scaler else None,
    }

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------
    np.save(out_dir / "X_clean.npy", X)
    np.save(out_dir / "y_clean.npy", y)

    if scaler is not None:
        with open(out_dir / "scaler.json", "w", encoding="utf-8") as f:
            json.dump(scaler, f, indent=2)

    with open(out_dir / "feature_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log_event(
        f"Features prepared: {report['final_samples']} samples, "
        f"{report['n_features']} features"
    )

    return report
