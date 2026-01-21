# ============================================================
# core/dataset_builder.py
# ============================================================

import json
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.ops import unary_union


# ------------------------------------------------------------
# OPTIONAL IMPORTS
# ------------------------------------------------------------
try:
    from logger import log_event
except Exception:
    def log_event(msg):
        print(msg)

try:
    from validators import validate_raster_pair
except Exception:
    validate_raster_pair = None


# ---------------- CONFIG (LOCKED) ----------------
BARE_CLASS_ID = 0
BARE_LABEL = "bare_land"
BARE_NDVI_MIN = 0.0
BARE_NDVI_MAX = 0.2
BARE_MAX_SAMPLES = 300

USER_MIN_SAMPLES = 30
USER_MAX_SAMPLES = 500

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def build_dataset(
    stack_path: str,
    ndvi_path: str,
    shapefile_info: list,
    out_dir: str
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

    X_all, y_all = [], []
    report = {
        "bare_land": {},
        "user_classes": {},
        "warnings": []
    }

    log_event("Building training dataset")

    # --------------------------------------------------------
    # INPUT PATH VALIDATION
    # --------------------------------------------------------
    stack_path = Path(stack_path)
    ndvi_path = Path(ndvi_path)

    if not stack_path.exists():
        raise FileNotFoundError(f"Stack raster not found: {stack_path}")

    if not ndvi_path.exists():
        raise FileNotFoundError(f"NDVI raster not found: {ndvi_path}")

    with rasterio.open(stack_path) as stack_ds, rasterio.open(ndvi_path) as ndvi_ds:

        if validate_raster_pair is not None:
            validate_raster_pair(stack_ds, ndvi_ds)

        if stack_ds.crs is None or ndvi_ds.crs is None:
            raise ValueError("Missing CRS in stack or NDVI")

        if stack_ds.crs != ndvi_ds.crs:
            raise ValueError("CRS mismatch between stack and NDVI")

        if stack_ds.transform != ndvi_ds.transform:
            raise ValueError("Transform mismatch between stack and NDVI")

        bands = stack_ds.count
        height, width = stack_ds.height, stack_ds.width
        pixel_area = abs(stack_ds.res[0] * stack_ds.res[1])

        if bands < 1:
            raise ValueError("Stack raster has no bands")

        # ====================================================
        # 1️⃣ USER CLASSES
        # ====================================================
        all_user_geoms = []

        for cls in shapefile_info:
            zip_path = Path(cls["zip_path"])
            label = cls["label"]
            class_id = int(cls["class_id"])

            if class_id <= 0:
                raise ValueError("class_id must be >= 1")

            if not zip_path.exists():
                report["warnings"].append(f"Missing ZIP skipped: {zip_path}")
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(tmpdir)

                shp_files = list(Path(tmpdir).glob("*.shp"))
                if len(shp_files) != 1:
                    raise ValueError(f"ZIP must contain exactly one .shp: {zip_path}")

                gdf = gpd.read_file(shp_files[0])

            if gdf.empty:
                report["warnings"].append(f"Empty shapefile skipped: {label}")
                continue

            if gdf.crs is None:
                raise ValueError(f"Missing CRS for class '{label}'")

            if gdf.crs != stack_ds.crs:
                raise ValueError(f"CRS mismatch for class '{label}'")

            all_user_geoms.append(unary_union(gdf.geometry))

            X_cls, y_cls = [], []

            areas_px = (gdf.geometry.area / pixel_area).values
            median_area_px = np.median(areas_px)

            if not np.isfinite(median_area_px) or median_area_px <= 0:
                report["warnings"].append(f"Invalid geometry area for '{label}'")
                continue

            auto_factor = np.clip(median_area_px ** -0.5, 0.01, 0.2)

            for geom, area_px in zip(gdf.geometry, areas_px):
                if area_px <= 0 or not np.isfinite(area_px):
                    continue

                base = np.sqrt(area_px)
                n_samples = int(base * auto_factor * 100)
                n_samples = int(np.clip(n_samples, USER_MIN_SAMPLES, USER_MAX_SAMPLES))

                try:
                    img, _ = mask(stack_ds, [geom], crop=True)
                    ndv, _ = mask(ndvi_ds, [geom], crop=True)
                except ValueError:
                    continue

                img = img.astype(np.float32)
                ndv = ndv.astype(np.float32)

                _, H, W = img.shape
                if H < 3 or W < 3:
                    continue

                valid = np.ones((H, W), dtype=bool)
                valid[[0, -1], :] = False
                valid[:, [0, -1]] = False

                coords = np.column_stack(np.where(valid))
                if coords.size == 0:
                    continue

                choose = min(n_samples, len(coords))
                idx = np.random.choice(len(coords), choose, replace=False)

                for r, c in coords[idx]:
                    px = img[:, r, c]
                    nd = ndv[0, r, c]
                    if np.isfinite(px).all() and np.isfinite(nd):
                        X_cls.append(np.concatenate([px, [nd]]))
                        y_cls.append(class_id)

            if X_cls:
                X_all.append(np.array(X_cls, dtype=np.float32))
                y_all.append(np.array(y_cls, dtype=np.int64))
                report["user_classes"][label] = {
                    "class_id": class_id,
                    "samples": len(X_cls)
                }

        # ====================================================
        # 2️⃣ BARE LAND
        # ====================================================
        ndvi = ndvi_ds.read(1).astype(np.float32)
        stack = stack_ds.read().astype(np.float32)

        bare_mask = (ndvi >= BARE_NDVI_MIN) & (ndvi <= BARE_NDVI_MAX)

        if all_user_geoms:
            union_geom = unary_union(all_user_geoms)
            geom_mask = geometry_mask(
                [union_geom],
                transform=stack_ds.transform,
                invert=True,
                out_shape=(height, width)
            )
            bare_mask &= geom_mask

        coords = np.column_stack(np.where(bare_mask))
        if coords.size > 0:
            choose = min(BARE_MAX_SAMPLES, len(coords))
            idx = np.random.choice(len(coords), choose, replace=False)

            X_bare, y_bare = [], []
            for r, c in coords[idx]:
                px = stack[:, r, c]
                nd = ndvi[r, c]
                if np.isfinite(px).all() and np.isfinite(nd):
                    X_bare.append(np.concatenate([px, [nd]]))
                    y_bare.append(BARE_CLASS_ID)

            if X_bare:
                X_all.append(np.array(X_bare, dtype=np.float32))
                y_all.append(np.array(y_bare, dtype=np.int64))

            report["bare_land"] = {
                "ndvi_range": [BARE_NDVI_MIN, BARE_NDVI_MAX],
                "samples": len(X_bare),
                "max_cap": BARE_MAX_SAMPLES
            }

        # ====================================================
        # 3️⃣ FINALIZE
        # ====================================================
        if not X_all:
            raise RuntimeError("No samples generated")

        X = np.vstack(X_all)
        y = np.concatenate(y_all)

        if X.shape[0] != y.shape[0]:
            raise RuntimeError("X/y size mismatch")

        np.save(out_dir / "X.npy", X)
        np.save(out_dir / "y.npy", y)

        class_map = {0: BARE_LABEL}
        for cls in shapefile_info:
            class_map[int(cls["class_id"])] = cls["label"]

        with open(out_dir / "class_map.json", "w") as f:
            json.dump(class_map, f, indent=2)

        with open(out_dir / "dataset_report.json", "w") as f:
            json.dump(report, f, indent=2)

        log_event(f"Dataset built: {X.shape[0]} samples")

        return report
