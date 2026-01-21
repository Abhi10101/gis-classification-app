# ============================================================
# cleanup.py
# ============================================================
# PURPOSE:
# - Safe, explicit cleanup utility
# - Recursive delete for files / directories
# - Idempotent behavior
#
# DESIGN RULES:
# - NO silent fallback
# - NO retries
# - NO background threads
# - NO UI dependency
#
# Python: 3.8 / 3.9 compatible
# ============================================================

from pathlib import Path
import shutil


def safe_cleanup(path, *, force: bool = False) -> bool:
    """
    Safely delete a file or directory.

    Args:
        path: file or directory path
        force: must be True to allow directory deletion

    Returns:
        True  -> deleted successfully OR already not present
        False -> invalid input (None)

    Raises:
        RuntimeError on unsafe or unknown path types
    """

    if path is None:
        return False

    p = Path(path).resolve()

    # Safety guard: never allow root or very short paths
    if len(p.parts) <= 2:
        raise RuntimeError(f"Refusing to delete unsafe path: {p}")

    if not p.exists():
        return True

    if p.is_file():
        p.unlink()
        return True

    if p.is_dir():
        if not force:
            raise RuntimeError(f"Directory deletion requires force=True: {p}")
        shutil.rmtree(p)
        return True

    raise RuntimeError(f"Unknown path type: {p}")
