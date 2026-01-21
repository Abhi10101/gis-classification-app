# ============================================================
# core/train_xgb.py
# ============================================================

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


def train_xgboost(
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
        raise ValueError("Empty dataset: X has zero samples")

    # CRITICAL for XGB
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values found in X")

    classes, counts = np.unique(y, return_counts=True)

    if len(classes) < 2:
        raise ValueError("XGBoost requires at least 2 classes")

    if np.any(counts < 2):
        raise ValueError(
            f"Each class must have >=2 samples for stratified split. "
            f"Counts: {dict(zip(classes.tolist(), counts.tolist()))}"
        )

    n_classes = len(classes)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    if len(X_va) == 0:
        raise RuntimeError("Validation split resulted in empty set")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_tr),
        y=y_tr
    )
    class_weight = dict(zip(np.unique(y_tr).tolist(), weights.tolist()))
    sample_weight = np.array(
        [class_weight[int(i)] for i in y_tr],
        dtype=np.float32
    )

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=600,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        gamma=0.1,
        tree_method="auto",
        n_jobs=1,
        random_state=random_state
    )

    model.fit(
        X_tr,
        y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_va, y_va)],
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        verbose=False
    )

    y_pred = model.predict(X_va)
    report_txt = classification_report(
        y_va,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report = {
        "model": "XGBoost",
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_classes": int(n_classes),
        "class_weight": class_weight,
        "validation_report": report_txt,
        "params": model.get_params()
    }

    model.save_model(out_dir / "xgb_model.json")

    with open(out_dir / "xgb_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
