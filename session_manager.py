# ============================================================
# session_manager.py
# ============================================================
# PURPOSE:
# - Manage per-user session lifecycle
# - Create isolated temp directories
# - Guarantee cleanup after download or failure
#
# DESIGN RULES:
# - NO inference logic
# - NO training logic
# - NO UI dependency
# - SAFE for Streamlit / HF free servers
#
# Python: 3.8 / 3.9 compatible
# ============================================================

from pathlib import Path
import tempfile
import shutil
import uuid


class SessionManager:
    """
    Manages a single user session directory.
    """

    def __init__(self, prefix: str = "gis_run_"):
        self._prefix = prefix
        self._session_id = uuid.uuid4().hex[:8]
        self._base_dir: Path | None = None
        self._downloaded: bool = False

    # --------------------------------------------------------
    # CREATE SESSION
    # --------------------------------------------------------
    def create(self) -> Path:
        """
        Create a new temporary session directory.
        """
        if self._base_dir is not None:
            return self._base_dir

        self._base_dir = Path(
            tempfile.mkdtemp(prefix=f"{self._prefix}{self._session_id}_")
        ).resolve()

        return self._base_dir

    # --------------------------------------------------------
    # RESOLVE PATH INSIDE SESSION
    # --------------------------------------------------------
    def path(self, *parts) -> Path:
        """
        Resolve a path inside the session directory.
        """
        if self._base_dir is None:
            raise RuntimeError("Session not created yet")

        return self._base_dir.joinpath(*parts).resolve()

    # --------------------------------------------------------
    # MARK DOWNLOAD COMPLETE
    # --------------------------------------------------------
    def mark_downloaded(self) -> None:
        """
        Mark that outputs were successfully downloaded.
        """
        self._downloaded = True

    # --------------------------------------------------------
    # CLEANUP SESSION
    # --------------------------------------------------------
    def cleanup(self, force: bool = False) -> None:
        """
        Cleanup session directory.

        Parameters:
        - force=True : always delete
        - force=False: delete only if downloaded
        """
        if self._base_dir is None:
            return

        if not force and not self._downloaded:
            return

        base = self._base_dir

        # Safety: never delete root / home
        if base == Path("/") or base == Path.home():
            raise RuntimeError(f"Unsafe cleanup path: {base}")

        shutil.rmtree(base)

        self._base_dir = None
        self._downloaded = False

    # --------------------------------------------------------
    # CHECK SESSION EXISTS
    # --------------------------------------------------------
    def exists(self) -> bool:
        """
        Check if session directory exists.
        """
        return self._base_dir is not None and self._base_dir.exists()
