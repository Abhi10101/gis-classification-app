# ============================================================
# run_tracker.py
# ============================================================
# PURPOSE:
# - Lightweight session-based MLOps tracker
# - Tracks timings, sizes, and health
# - Exports ONE CSV per session
#
# DESIGN RULES:
# - NO persistence
# - NO networking
# - NO UI dependency
# - FREE server safe
#
# Python: 3.8 / 3.9 compatible
# ============================================================

from pathlib import Path
from datetime import datetime
import csv
import time

# Optional memory tracking (safe if missing)
try:
    import psutil
except Exception:
    psutil = None


class RunTracker:
    """
    Tracks a single session run.
    """

    def __init__(self, session_id: str):
        self.session_id = str(session_id)

        self.start_time = None
        self.train_times = {}
        self.inference_time = None
        self.dataset_size = None
        self.accuracy = None
        self.cleanup_status = None
        self.peak_memory_mb = None

    # --------------------------------------------------------
    # START SESSION
    # --------------------------------------------------------
    def start(self) -> None:
        self.start_time = time.time()

        if psutil is not None:
            try:
                proc = psutil.Process()
                self.peak_memory_mb = proc.memory_info().rss / (1024 * 1024)
            except Exception:
                self.peak_memory_mb = None

    # --------------------------------------------------------
    # LOG TRAINING
    # --------------------------------------------------------
    def log_train_time(self, model_name: str, seconds: float) -> None:
        self.train_times[str(model_name)] = float(seconds)

    # --------------------------------------------------------
    # LOG INFERENCE
    # --------------------------------------------------------
    def log_inference_time(self, seconds: float) -> None:
        self.inference_time = float(seconds)

    # --------------------------------------------------------
    # DATASET SIZE
    # --------------------------------------------------------
    def set_dataset_size(self, n_samples: int) -> None:
        self.dataset_size = int(n_samples)

    # --------------------------------------------------------
    # ACCURACY (OPTIONAL)
    # --------------------------------------------------------
    def set_accuracy(self, value: float) -> None:
        self.accuracy = float(value)

    # --------------------------------------------------------
    # CLEANUP STATUS
    # --------------------------------------------------------
    def set_cleanup_status(self, status: str) -> None:
        self.cleanup_status = str(status)

    # --------------------------------------------------------
    # EXPORT CSV
    # --------------------------------------------------------
    def export_csv(self, out_path) -> Path:
        """
        Export run metrics as CSV.
        """
        if self.start_time is None:
            raise RuntimeError("RunTracker.start() was not called")

        out_path = Path(out_path)

        if out_path.exists() and out_path.is_dir():
            raise ValueError(f"CSV output path is a directory: {out_path}")

        if not out_path.parent.exists():
            raise FileNotFoundError(
                f"Parent directory does not exist: {out_path.parent}"
            )

        rows = {
            "session_id": self.session_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "dataset_size": self.dataset_size,
            "train_time_lr": self.train_times.get("lr"),
            "train_time_rf": self.train_times.get("rf"),
            "train_time_xgb": self.train_times.get("xgb"),
            "inference_time": self.inference_time,
            "accuracy": self.accuracy,
            "peak_memory_mb": self.peak_memory_mb,
            "cleanup_status": self.cleanup_status,
        }

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows.keys()))
            writer.writeheader()
            writer.writerow(rows)

        return out_path
