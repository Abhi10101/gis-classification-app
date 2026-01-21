# ============================================================
# core/train_rf.py
# ============================================================

import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


def train_random_forest(
    X_path: str,
    y_path: str,
    out_dir: str,
    test_size: float = 0.25,
    random_state: int = 42
):
    """
    DO NOT CHANGE SIGNATURE
    """

    # --------------------------------------------------------
    # STRICT PATH OWNERSHIP (NO mkdir HERE)
    # --------------------------------------------------------
    out_dir = Path(out_dir)

    if not out_dir.exists():
        raise FileNotFoundError(
            f"Output directory does not exist (must be created by app.py): {out_dir}"
        )

    if not out_dir.is_dir():
        raise NotADirectoryError(f"out_dir is not a directory: {out_dir}")

    X_path = Path(X_path)
    y_path = Path(y_path)

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError("X or y file not found")

    X = np.load(X_path)
    y = np.load(y_path)

    if len(X) == 0:
        raise ValueError("Empty dataset")

    # Protect against NaN / Inf
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values found in X")

    classes, counts = np.unique(y, return_counts=True)

    if len(classes) < 2:
        raise ValueError("RandomForest requires at least 2 classes")

    if np.any(counts < 2):
        raise ValueError(
            f"Each class must have >=2 samples. "
            f"Counts: {dict(zip(classes.tolist(), counts.tolist()))}"
        )

    X_tr, X_va, y_tr, y_va = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    if len(X_va) == 0:
        raise RuntimeError("Validation split resulted in empty set")

    classes_tr = np.unique(y_tr)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_tr,
        y=y_tr
    )
    class_weight = dict(zip(classes_tr.tolist(), weights.tolist()))

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=5,
        n_jobs=1,   # server-safe
        class_weight=class_weight,
        random_state=random_state
    )

    rf.fit(X_tr, y_tr)

    report = {
        "model": "RandomForest",
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "class_weight": class_weight,
        "validation_report": classification_report(
            y_va, rf.predict(X_va), output_dict=True, zero_division=0
        ),
        "params": rf.get_params()
    }

    joblib.dump(rf, out_dir / "rf_model.joblib")

    with open(out_dir / "rf_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
