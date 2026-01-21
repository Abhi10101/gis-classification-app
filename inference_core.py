# ============================================================
# inference/inference_core.py — OPTIMIZED & SAFE
# ============================================================

from pathlib import Path
from typing import Optional, Dict, Callable

import numpy as np
import rasterio
from rasterio.windows import Window
import joblib
import xgboost as xgb

from ensemble import ensemble_probabilities


# ------------------------------------------------------------
# Optional imports
# ------------------------------------------------------------
try:
    from logger import log_event
except Exception:
    def log_event(msg: str):
        pass

try:
    from validators import validate_raster_pair
except Exception:
    validate_raster_pair = None


# ------------------------------------------------------------
# INTERNALS
# ------------------------------------------------------------
_PROGRESS_LOG_EVERY = 10
_progress_callback: Optional[Callable[[float, str], None]] = None


def set_progress_callback(cb: Optional[Callable[[float, str], None]]):
    global _progress_callback
    _progress_callback = cb


def _emit_progress(p: float, msg: str):
    if _progress_callback:
        _progress_callback(float(min(max(p, 0), 1)), msg)


# ------------------------------------------------------------
# MAIN ENTRY (SIGNATURE UNCHANGED)
# ------------------------------------------------------------
def run_inference(
    stack_path: str,
    ndvi_path: str,
    model_dir: str,
    out_dir: str,
    tile_size: int = 512,
    ensemble_weights: Optional[Dict] = None
):

    stack_path = Path(stack_path)
    ndvi_path = Path(ndvi_path)
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)

    # ---------------- VALIDATION ----------------
    for p, n in [
        (stack_path, "Stack"),
        (ndvi_path, "NDVI"),
        (model_dir, "Model dir"),
        (out_dir, "Output dir"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{n} not found: {p}")

    log_event("Inference started")
    _emit_progress(0.0, "Initializing")

    # ---------------- LOAD MODELS ONCE ----------------
    rf = joblib.load(model_dir / "rf_model.joblib")

    xgb_model = xgb.XGBClassifier(
        tree_method="hist",
        predictor="cpu_predictor"
    )
    xgb_model.load_model(model_dir / "xgb_model.json")

    lr_path = model_dir / "lr_model.joblib"
    lr = joblib.load(lr_path) if lr_path.exists() else None

    models = [rf, xgb_model] + ([lr] if lr else [])

    # ---------------- ENSEMBLE WEIGHTS ----------------
    ensemble_weights = ensemble_weights or {}
    weights = np.array(
        [
            ensemble_weights.get("rf", 1.0),
            ensemble_weights.get("xgb", 1.0),
            ensemble_weights.get("lr", 1.0) if lr else 0.0,
        ],
        dtype=np.float32
    )
    weights = weights[:len(models)]
    weights /= weights.sum()

    # ---------------- OPEN RASTERS ----------------
    with rasterio.open(stack_path) as stack_ds, rasterio.open(ndvi_path) as ndvi_ds:

        if validate_raster_pair:
            validate_raster_pair(stack_ds, ndvi_ds)

        if stack_ds.crs != ndvi_ds.crs or stack_ds.transform != ndvi_ds.transform:
            raise ValueError("Raster alignment mismatch")

        H, W = stack_ds.height, stack_ds.width
        B = stack_ds.count

        profile = stack_ds.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, nodata=255, compress="lzw")

        conf_profile = profile.copy()
        conf_profile.update(dtype=rasterio.float32, nodata=None)

        out_class = out_dir / "predicted_class.tif"
        out_conf = out_dir / "prediction_confidence.tif"

        tiles_y = (H + tile_size - 1) // tile_size
        tiles_x = (W + tile_size - 1) // tile_size
        total_tiles = tiles_x * tiles_y
        done = 0

        with rasterio.open(out_class, "w", **profile) as cls_dst, \
             rasterio.open(out_conf, "w", **conf_profile) as conf_dst:

            for r in range(0, H, tile_size):
                for c in range(0, W, tile_size):

                    done += 1
                    h = min(tile_size, H - r)
                    w = min(tile_size, W - c)
                    win = Window(c, r, w, h)

                    stack = stack_ds.read(window=win).astype(np.float32)
                    ndvi = ndvi_ds.read(1, window=win).astype(np.float32)

                    X = np.concatenate(
                        [stack.reshape(B, -1).T, ndvi.reshape(-1, 1)],
                        axis=1
                    )

                    valid = np.isfinite(X).all(axis=1)

                    cls = np.full(h * w, 255, dtype=np.uint8)
                    conf = np.zeros(h * w, dtype=np.float32)

                    if valid.any():
                        Xv = X[valid]

                        # ---- FAST PATH: RF + XGB agree ----
                        rf_p = rf.predict(Xv)
                        xgb_p = xgb_model.predict(Xv)

                        agree = rf_p == xgb_p

                        cls_valid = np.empty(len(Xv), dtype=np.uint8)
                        conf_valid = np.empty(len(Xv), dtype=np.float32)

                        if agree.all():
                            cls_valid[:] = rf_p
                            conf_valid[:] = rf.predict_proba(Xv).max(axis=1)
                        else:
                            probs = [
                                rf.predict_proba(Xv),
                                xgb_model.predict_proba(Xv),
                            ]
                            if lr:
                                probs.append(lr.predict_proba(Xv))

                            ens = ensemble_probabilities(probs, weights)
                            cls_valid[:] = ens.argmax(axis=1)
                            conf_valid[:] = ens.max(axis=1)

                        cls[valid] = cls_valid
                        conf[valid] = conf_valid

                    cls_dst.write(cls.reshape(h, w), 1, window=win)
                    conf_dst.write(conf.reshape(h, w), 1, window=win)

                    if done % _PROGRESS_LOG_EVERY == 0 or done == total_tiles:
                        frac = done / total_tiles
                        _emit_progress(frac, f"Inference {done}/{total_tiles}")

    log_event("Inference completed")
    _emit_progress(1.0, "Inference completed")

    return {
        "status": "success",
        "outputs": {
            "class_map": str(out_class),
            "confidence_map": str(out_conf),
        }
    }
