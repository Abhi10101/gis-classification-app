# ============================================================
# ndvi_rules.py
# ============================================================

from pathlib import Path
import numpy as np
import rasterio


# ------------------------------------------------------------
# OPTIONAL IMPORTS – safe if missing
# ------------------------------------------------------------
try:
    from logger import log_event
except Exception:
    def log_event(msg):
        pass

try:
    from validators import validate_raster_triple
except Exception:
    validate_raster_triple = None


# ---------------- LOCKED RULES ----------------
BARE_CLASS_ID = 0
NDVI_MIN = 0.0
NDVI_MAX = 0.2
NODATA_VALUE = 255


def apply_ndvi_rules(
    class_map_path: str,
    confidence_map_path: str,
    ndvi_path: str,
    out_path: str,
    confidence_threshold: float = 0.45
):
    """
    DO NOT CHANGE SIGNATURE
    """

    # --------------------------------------------------------
    # STRICT PATH VALIDATION (NO mkdir HERE)
    # --------------------------------------------------------
    class_map_path = Path(class_map_path)
    confidence_map_path = Path(confidence_map_path)
    ndvi_path = Path(ndvi_path)
    out_path = Path(out_path)

    if not class_map_path.exists():
        raise FileNotFoundError(f"class_map_path not found: {class_map_path}")

    if not confidence_map_path.exists():
        raise FileNotFoundError(f"confidence_map_path not found: {confidence_map_path}")

    if not ndvi_path.exists():
        raise FileNotFoundError(f"ndvi_path not found: {ndvi_path}")

    if not out_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist (must be created by app.py): "
            f"{out_path.parent}"
        )

    if not out_path.parent.is_dir():
        raise NotADirectoryError(
            f"Output parent is not a directory: {out_path.parent}"
        )

    log_event("Applying NDVI override rules")

    with rasterio.open(class_map_path) as cls_ds, \
         rasterio.open(confidence_map_path) as conf_ds, \
         rasterio.open(ndvi_path) as ndvi_ds:

        # FUTURE: centralized triple-raster validation
        if validate_raster_triple is not None:
            validate_raster_triple(cls_ds, conf_ds, ndvi_ds)

        # --------------------------------------------------------
        # Validation (strict)
        # --------------------------------------------------------
        if cls_ds.crs is None or ndvi_ds.crs is None or conf_ds.crs is None:
            raise ValueError("Missing CRS in one or more rasters")

        if cls_ds.crs != ndvi_ds.crs:
            raise ValueError("CRS mismatch: class map vs NDVI")

        if cls_ds.crs != conf_ds.crs:
            raise ValueError("CRS mismatch: class map vs confidence")

        if cls_ds.transform != ndvi_ds.transform:
            raise ValueError("Transform mismatch: class map vs NDVI")

        if cls_ds.transform != conf_ds.transform:
            raise ValueError("Transform mismatch: class map vs confidence")

        if cls_ds.shape != ndvi_ds.shape:
            raise ValueError("Shape mismatch: class map vs NDVI")

        if cls_ds.shape != conf_ds.shape:
            raise ValueError("Shape mismatch: class map vs confidence")

        # --------------------------------------------------------
        # Read rasters
        # --------------------------------------------------------
        cls = cls_ds.read(1).astype(np.int64, copy=False)
        conf = conf_ds.read(1).astype(np.float32, copy=False)
        ndvi = ndvi_ds.read(1).astype(np.float32, copy=False)

        # --------------------------------------------------------
        # Validity masks
        # --------------------------------------------------------
        valid_cls = cls != NODATA_VALUE
        valid_conf = np.isfinite(conf)
        valid_ndvi = np.isfinite(ndvi)

        valid_mask = valid_cls & valid_conf & valid_ndvi

        if not np.any(valid_mask):
            log_event("No valid pixels for NDVI rules")

        # --------------------------------------------------------
        # NDVI bare logic (LOCKED)
        # --------------------------------------------------------
        if not np.isfinite(confidence_threshold) or confidence_threshold < 0:
            raise ValueError("confidence_threshold must be finite and >= 0")

        bare_mask = (ndvi >= NDVI_MIN) & (ndvi <= NDVI_MAX)
        low_conf = conf < confidence_threshold

        override_mask = valid_mask & bare_mask & low_conf

        corrected = cls.copy()
        corrected[override_mask] = BARE_CLASS_ID

        # --------------------------------------------------------
        # Save output
        # --------------------------------------------------------
        profile = cls_ds.profile
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw",
            nodata=NODATA_VALUE
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(corrected.astype(np.uint8), 1)

    log_event(
        f"NDVI override applied: {int(override_mask.sum())} pixels"
    )

    return {
        "status": "success",
        "overridden_pixels": int(override_mask.sum()),
        "confidence_threshold": confidence_threshold,
        "ndvi_range": [NDVI_MIN, NDVI_MAX],
        "output": str(out_path)
    }
