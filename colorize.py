# ============================================================
# colorize.py
# ============================================================

import json
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
        print(msg)

try:
    from validators import validate_raster_path
except Exception:
    validate_raster_path = None


NODATA_VALUE = 255


def colorize_class_map(
    class_map_path: str,
    color_map_json: str,
    out_path: str
):
    """
    DO NOT CHANGE SIGNATURE
    """

    # --------------------------------------------------------
    # STRICT PATH VALIDATION (NO mkdir HERE)
    # --------------------------------------------------------
    class_map_path = Path(class_map_path)
    out_path = Path(out_path)
    color_map_json = Path(color_map_json)

    if not class_map_path.exists():
        raise FileNotFoundError(f"class_map_path not found: {class_map_path}")

    if not color_map_json.exists():
        raise FileNotFoundError(f"color_map_json not found: {color_map_json}")

    if not out_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist (must be created by app.py): "
            f"{out_path.parent}"
        )

    if not out_path.parent.is_dir():
        raise NotADirectoryError(
            f"Output parent is not a directory: {out_path.parent}"
        )

    log_event("Colorizing class map")

    # --------------------------------------------------------
    # Raster validation
    # --------------------------------------------------------
    if validate_raster_path is not None:
        validate_raster_path(class_map_path)

    # --------------------------------------------------------
    # Load color map
    # --------------------------------------------------------
    with open(color_map_json, "r") as f:
        raw_map = json.load(f)

    if not isinstance(raw_map, dict):
        raise ValueError("color_map_json must be a dict")

    color_map = {}
    for k, v in raw_map.items():
        try:
            color_map[int(k)] = list(v)
        except Exception:
            raise ValueError(f"Invalid color map entry: {k} -> {v}")

    # --------------------------------------------------------
    # Read class map raster
    # --------------------------------------------------------
    with rasterio.open(class_map_path) as src:
        cls = src.read(1)

        if cls.ndim != 2:
            raise ValueError("Class map must be single-band")

        profile = src.profile

        # Handle nodata in source
        src_nodata = src.nodata
        if src_nodata is not None:
            cls = np.where(cls == src_nodata, NODATA_VALUE, cls)

        # ----------------------------------------------------
        # Validate class coverage
        # ----------------------------------------------------
        unique_classes = set(np.unique(cls))
        unique_classes.discard(NODATA_VALUE)

        missing = unique_classes - set(color_map.keys())
        if missing:
            raise ValueError(
                f"Missing colors for class IDs: {sorted(missing)}"
            )

        # ----------------------------------------------------
        # Prepare RGB output
        # ----------------------------------------------------
        profile.update(
            count=3,
            dtype=rasterio.uint8,
            nodata=0,
            compress="lzw"
        )

        h, w = cls.shape
        rgb = np.zeros((3, h, w), dtype=np.uint8)

        # ----------------------------------------------------
        # Apply colors
        # ----------------------------------------------------
        for class_id, color in color_map.items():
            if (
                not isinstance(color, (list, tuple)) or
                len(color) != 3 or
                any((c < 0 or c > 255) for c in color)
            ):
                raise ValueError(
                    f"Invalid RGB color for class {class_id}: {color}"
                )

            mask = cls == class_id
            if mask.any():
                rgb[0][mask] = color[0]
                rgb[1][mask] = color[1]
                rgb[2][mask] = color[2]

        # Preserve nodata pixels
        nodata_mask = cls == NODATA_VALUE
        if nodata_mask.any():
            rgb[:, nodata_mask] = 0

        # ----------------------------------------------------
        # Write output
        # ----------------------------------------------------
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(rgb)

    log_event(f"Colorized raster written: {out_path}")

    return {
        "status": "success",
        "output": str(out_path),
        "classes_colored": sorted(color_map.keys())
    }
