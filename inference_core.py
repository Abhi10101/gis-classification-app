# ============================================================
# inference/inference_core.py
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
# Optional imports (explicit, no silent logic change)
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
# INTERNAL CONSTANTS
# ------------------------------------------------------------
_PROGRESS_LOG_EVERY = 10


# ------------------------------------------------------------
# PROGRESS CALLBACK
# ------------------------------------------------------------
_progress_callback: Optional[Callable[[float, str], None]] = None


def set_progress_callback(cb: Optional[Callable[[float, str], None]]):
    """UI hook: cb(progress[0–1], message)"""
    global _progress_callback
    _progress_callback = cb


def _emit_progress(progress: float, message: str):
    if _progress_callback is None:
        return
    p = min(max(float(progress), 0.0), 1.0)
    _progress_callback(p, message)


# ------------------------------------------------------------
# MAIN ENTRY (DO NOT CHANGE SIGNATURE)
# ------------------------------------------------------------
def run_inference(
    stack_path: str,
    ndvi_path: str,
    model_dir: str,
    out_dir: str,
    tile_size: int = 512,
    ensemble_weights: Optional[Dict] = None
):
    """
    Inference core
    Python 3.8 / 3.9 compatible
    """

    stack_path = Path(stack_path)
    ndvi_path = Path(ndvi_path)
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)

    # --------------------------------------------------------
    # STRICT PATH VALIDATION
    # --------------------------------------------------------
    if not stack_path.exists():
        raise FileNotFoundError(f"Stack raster not found: {stack_path}")

    if not ndvi_path.exists():
        raise FileNotFoundError(f"NDVI raster not found: {ndvi_path}")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not out_dir.exists() or not out_dir.is_dir():
        raise FileNotFoundError(f"Output directory missing: {out_dir}")

    log_event("Inference started")
    _emit_progress(0.0, "Initializing inference")

    # --------------------------------------------------------
    # LOAD MODELS (ONCE)
    # --------------------------------------------------------
    rf = joblib.load(model_dir / "rf_model.joblib")

    xgb_model = xgb.XGBClassifier(
        tree_method="hist",
        predictor="cpu_predictor"
    )
    xgb_model.load_model(model_dir / "xgb_model.json")

    lr_path = model_dir / "lr_model.joblib"
    lr = joblib.load(lr_path) if lr_path.exists() else None

    models = [rf, xgb_model]
    if lr is not None:
        models.append(lr)

    # --------------------------------------------------------
    # ENSEMBLE WEIGHTS (STRICT)
    # --------------------------------------------------------
    if ensemble_weights is None:
        ensemble_weights = {}

    weights = []
    weights.append(float(ensemble_weights.get("rf", 1.0)))
    weights.append(float(ensemble_weights.get("xgb", 1.0)))
    if lr is not None:
        weights.append(float(ensemble_weights.get("lr", 1.0)))

    weight_arr = np.asarray(weights, dtype=np.float32)

    if not np.isfinite(weight_arr).all() or np.any(weight_arr < 0):
        raise ValueError("Invalid ensemble weights")

    if weight_arr.sum() <= 0:
        raise ValueError("Ensemble weights sum must be > 0")

    weight_arr /= weight_arr.sum()

    # --------------------------------------------------------
    # OPEN RASTERS
    # --------------------------------------------------------
    with rasterio.open(stack_path) as stack_ds, rasterio.open(ndvi_path) as ndvi_ds:

        if validate_raster_pair is not None:
            validate_raster_pair(stack_ds, ndvi_ds)

        if stack_ds.crs != ndvi_ds.crs:
            raise ValueError("CRS mismatch")

        if stack_ds.transform != ndvi_ds.transform:
            raise ValueError("Transform mismatch")

        height, width = stack_ds.height, stack_ds.width
        bands = stack_ds.count

        profile = stack_ds.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, nodata=255, compress="lzw")

        conf_profile = profile.copy()
        conf_profile.update(dtype=rasterio.float32, nodata=None)

        out_class = out_dir / "predicted_class.tif"
        out_conf = out_dir / "prediction_confidence.tif"

        tiles_y = (height + tile_size - 1) // tile_size
        tiles_x = (width + tile_size - 1) // tile_size
        total_tiles = tiles_y * tiles_x
        processed = 0

        log_event(f"Total tiles: {total_tiles}")

        with rasterio.open(out_class, "w", **profile) as cls_dst, \
             rasterio.open(out_conf, "w", **conf_profile) as conf_dst:

            for row in range(0, height, tile_size):
                for col in range(0, width, tile_size):

                    processed += 1
                    h = min(tile_size, height - row)
                    w = min(tile_size, width - col)
                    window = Window(col, row, w, h)

                    stack = stack_ds.read(window=window).astype(np.float32)
                    ndvi = ndvi_ds.read(1, window=window).astype(np.float32)

                    X_img = stack.reshape(bands, -1).T
                    X = np.empty((X_img.shape[0], bands + 1), dtype=np.float32)
                    X[:, :-1] = X_img
                    X[:, -1] = ndvi.reshape(-1)

                    valid = np.isfinite(X).all(axis=1)

                    cls_flat = np.full(h * w, 255, dtype=np.uint8)
                    conf_flat = np.zeros(h * w, dtype=np.float32)

                    if valid.any():
                        Xv = X[valid]

                        probs = [m.predict_proba(Xv) for m in models]
                        preds = [p.argmax(axis=1) for p in probs]

                        first = preds[0]
                        if all(np.array_equal(first, p) for p in preds[1:]):
                            cls_flat[valid] = first.astype(np.uint8)
                            conf_flat[valid] = probs[0].max(axis=1)
                        else:
                            ens = ensemble_probabilities(probs, weights=weight_arr)
                            cls_flat[valid] = ens.argmax(axis=1).astype(np.uint8)
                            conf_flat[valid] = ens.max(axis=1)

                    cls_dst.write(cls_flat.reshape(h, w), 1, window=window)
                    conf_dst.write(conf_flat.reshape(h, w), 1, window=window)

                    if processed % _PROGRESS_LOG_EVERY == 0 or processed == total_tiles:
                        frac = processed / total_tiles
                        msg = f"Inference {processed}/{total_tiles} tiles"
                        log_event(msg)
                        _emit_progress(frac, msg)

    log_event("Inference completed")
    _emit_progress(1.0, "Inference completed")

    return {
        "status": "success",
        "outputs": {
            "class_map": str(out_class),
            "confidence_map": str(out_conf)
        }
    }
