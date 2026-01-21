# ============================================================
# validators.py
# ============================================================
# PURPOSE:
# - Centralized validation utilities
# - Input safety checks for rasters, features, and models
# - NO training
# - NO inference
# - NO file writing
#
# Python: 3.8 / 3.9 compatible
# ============================================================

import numpy as np
from pathlib import Path
import rasterio


# ------------------------------------------------------------
# BASIC FILE VALIDATION
# ------------------------------------------------------------

def validate_file_exists(path, name="file"):
    """
    Ensure file exists on disk.
    """
    if path is None:
        raise ValueError(f"{name} is None")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")

    return True


# ------------------------------------------------------------
# RASTER VALIDATION
# ------------------------------------------------------------

def validate_raster_path(path):
    """
    Validate raster file can be opened.
    """
    validate_file_exists(path, "raster")

    try:
        with rasterio.open(path) as ds:
            if ds.width <= 0 or ds.height <= 0:
                raise ValueError("Raster has invalid dimensions")

            if ds.count <= 0:
                raise ValueError("Raster has no bands")

            if ds.transform is None:
                raise ValueError("Raster missing transform")

    except Exception as e:
        raise RuntimeError(f"Invalid raster file: {path}") from e

    return True


def validate_raster_pair(stack_ds, ndvi_ds):
    """
    Validate that stack and NDVI rasters are compatible.
    Used in dataset_builder & inference_core.
    """
    if stack_ds.crs is None or ndvi_ds.crs is None:
        raise ValueError("Missing CRS in raster(s)")

    if stack_ds.crs != ndvi_ds.crs:
        raise ValueError("CRS mismatch between rasters")

    if stack_ds.transform != ndvi_ds.transform:
        raise ValueError("Transform mismatch between rasters")

    if stack_ds.width != ndvi_ds.width or stack_ds.height != ndvi_ds.height:
        raise ValueError("Raster dimension mismatch")

    if stack_ds.count <= 0:
        raise ValueError("Stack raster has no bands")

    return True


def validate_raster_triple(cls_ds, conf_ds, ndvi_ds):
    """
    Validate three rasters used together (NDVI rules).
    """
    if cls_ds.crs != conf_ds.crs or cls_ds.crs != ndvi_ds.crs:
        raise ValueError("CRS mismatch among rasters")

    if cls_ds.transform != conf_ds.transform or cls_ds.transform != ndvi_ds.transform:
        raise ValueError("Transform mismatch among rasters")

    if cls_ds.shape != conf_ds.shape or cls_ds.shape != ndvi_ds.shape:
        raise ValueError("Raster shape mismatch")

    return True


# ------------------------------------------------------------
# FEATURE MATRIX VALIDATION
# ------------------------------------------------------------

def validate_feature_matrix(X, y):
    """
    Validate feature matrix and labels.
    Used in feature_engineering.
    """
    if X is None or y is None:
        raise ValueError("X or y is None")

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")

    if X.ndim != 2:
        raise ValueError("X must be 2D array")

    if y.ndim != 1:
        raise ValueError("y must be 1D array")

    if len(X) != len(y):
        raise ValueError("X and y length mismatch")

    if len(X) == 0:
        raise ValueError("Empty feature matrix")

    if not np.isfinite(X).all():
        raise ValueError("Non-finite values found in X")

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values found in y")

    return True


# ------------------------------------------------------------
# INFERENCE PROBABILITY VALIDATION
# ------------------------------------------------------------

def validate_probabilities(prob_array):
    """
    Validate probability array from model.
    """
    if prob_array is None:
        raise ValueError("Probability array is None")

    if not isinstance(prob_array, np.ndarray):
        raise TypeError("Probability output must be numpy array")

    if prob_array.ndim != 2:
        raise ValueError("Probability array must be 2D")

    if not np.isfinite(prob_array).all():
        raise ValueError("Non-finite values in probability array")

    # fast row sum check (no keepdims)
    row_sums = prob_array.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("Invalid probability distribution (sum <= 0)")

    return True


# ------------------------------------------------------------
# UI / INPUT VALIDATION
# ------------------------------------------------------------

def validate_inputs(inputs: dict):
    """
    High-level UI input validation.
    Used in app.py.
    """
    if inputs is None or not isinstance(inputs, dict):
        return False, "Invalid input structure"

    if not inputs.get("stack_file"):
        return False, "Stack image not provided"

    if not inputs.get("ndvi_file"):
        return False, "NDVI image not provided"

    classes = inputs.get("classes", [])
    valid_classes = 0

    for cls in classes:
        if cls.get("label") and cls.get("zip"):
            valid_classes += 1

    if valid_classes == 0:
        return False, "At least one valid class (label + shapefile) is required"

    return True, "OK"
