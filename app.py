# ============================================================
# app.py — FINAL DEPLOY-SAFE VERSION (STREAMLIT CLOUD)
# ============================================================

import streamlit as st

# ------------------------------------------------------------
# PAGE CONFIG — ABSOLUTELY FIRST STREAMLIT CALL
# ------------------------------------------------------------
st.set_page_config(
    page_title="GIS Image Classification | Abhinandan Sood",
    layout="wide"
)

# ------------------------------------------------------------
# STANDARD LIBS
# ------------------------------------------------------------
from pathlib import Path
import json
import zipfile
import io
import gc
import yaml
import atexit

# ------------------------------------------------------------
# INTERNAL NON-UI MODULES (SAFE)
# ------------------------------------------------------------
from dataset_builder import build_dataset
from feature_engineering import prepare_features
from train_rf import train_random_forest
from train_xgb import train_xgboost
from train_lr import train_logistic_regression
from inference_core import run_inference, set_progress_callback
from ndvi_rules import apply_ndvi_rules
from colorize import colorize_class_map

from session_manager import SessionManager
from cleanup import safe_cleanup
from run_tracker import RunTracker
from mailer import send_csv_email

try:
    from validators import validate_inputs
except Exception:
    validate_inputs = None

try:
    from logger import log_event, init_logger
except Exception:
    def log_event(msg, level="INFO"):
        print(f"[{level}] {msg}")

    def init_logger(_):
        pass


# ------------------------------------------------------------
# LIMITS
# ------------------------------------------------------------
MAX_RASTER_MB = 3000
MAX_CLASS_ZIPS = 10


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
def load_config():
    cfg = Path("config.yaml")
    if not cfg.exists():
        return {}
    return yaml.safe_load(cfg.read_text()) or {}


def validate_email_config(cfg):
    for k in ("sender", "smtp_server", "smtp_port", "app_password"):
        if not cfg.get(k):
            raise ValueError(f"Missing email config key: {k}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    # 🔥 IMPORT UI MODULES HERE (CRITICAL)
    import ui_1_main
    import ui_2_docs

    # ---------------- SESSION STATE ----------------
    defaults = {
        "session": None,
        "tracker": None,
        "models_trained": False,
        "inference_done": False,
        "cleanup_done": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    # ---------------- FAILSAFE CLEANUP ----------------
    def _final_cleanup():
        sess = st.session_state.get("session")
        if sess and not st.session_state.get("cleanup_done"):
            try:
                sess.cleanup(force=True)
            except Exception:
                pass

    atexit.register(_final_cleanup)

    # ---------------- CONFIG ----------------
    config = load_config()
    inference_cfg = config.get("inference", {})
    ensemble_cfg = config.get("ensemble", {})
    ndvi_cfg = config.get("ndvi_rules", {})
    email_cfg = config.get("email", {})

    tile_size = int(inference_cfg.get("tile_size", 512))
    ensemble_weights = ensemble_cfg.get("weights")
    ndvi_conf_threshold = ndvi_cfg.get("confidence_threshold")

    # ---------------- UI ----------------
    tab_home, tab_docs = st.tabs(["🏠 Home", "📘 Documentation"])

    # =========================================================
    # HOME
    # =========================================================
    with tab_home:

        inputs = ui_1_main.render()

        st.divider()
        st.subheader("🧠 Model Training")

        if st.button("🚀 Train Models"):

            stack_file = inputs.get("stack_file")
            ndvi_file = inputs.get("ndvi_file")
            classes = inputs.get("classes", [])

            if not stack_file or not ndvi_file:
                st.error("Stack & NDVI rasters required")
                st.stop()

            session = SessionManager()
            out_dir = session.create()
            init_logger(out_dir)

            tracker = RunTracker(out_dir.name)
            tracker.start()

            st.session_state.update({
                "session": session,
                "tracker": tracker,
                "models_trained": False,
                "inference_done": False,
                "cleanup_done": False,
            })

            try:
                with st.status("Training models...", expanded=True):
                    progress = st.progress(0)

                    stack_path = session.path("stack.tif")
                    ndvi_path = session.path("ndvi.tif")
                    stack_path.write_bytes(stack_file.read())
                    ndvi_path.write_bytes(ndvi_file.read())

                    shapefile_info = []
                    color_map = {0: [165, 42, 42]}
                    cid = 1

                    for cls in classes:
                        zp = session.path(f"class_{cid}.zip")
                        zp.write_bytes(cls["zip"].read())
                        shapefile_info.append({
                            "zip_path": str(zp),
                            "label": cls["label"],
                            "class_id": cid
                        })
                        hex_c = cls["color"].lstrip("#")
                        color_map[cid] = [int(hex_c[i:i+2], 16) for i in (0, 2, 4)]
                        cid += 1

                    (session.path("class_color_map.json")
                     .write_text(json.dumps(color_map, indent=2)))

                    build_dataset(str(stack_path), str(ndvi_path), shapefile_info, str(out_dir))
                    progress.progress(30)

                    prepare_features(
                        str(session.path("X.npy")),
                        str(session.path("y.npy")),
                        str(out_dir)
                    )
                    progress.progress(60)

                    train_random_forest(str(session.path("X_clean.npy")),
                                        str(session.path("y_clean.npy")), str(out_dir))
                    train_xgboost(str(session.path("X_clean.npy")),
                                  str(session.path("y_clean.npy")), str(out_dir))
                    train_logistic_regression(str(session.path("X_clean.npy")),
                                              str(session.path("y_clean.npy")), str(out_dir))

                    progress.progress(100)

                st.session_state["models_trained"] = True
                st.success("Training completed")

            finally:
                gc.collect()

        # ---------------- INFERENCE ----------------
        st.divider()
        st.subheader("🛰️ Inference")

        if st.button("▶️ Run Inference", disabled=not st.session_state["models_trained"]):

            session = st.session_state["session"]

            with st.status("Running inference...", expanded=True):
                progress = st.progress(0)
                set_progress_callback(lambda p, m: progress.progress(p, m))

                run_inference(
                    str(session.path("stack.tif")),
                    str(session.path("ndvi.tif")),
                    str(session.path()),
                    str(session.path()),
                    tile_size,
                    ensemble_weights
                )

                apply_ndvi_rules(
                    str(session.path("predicted_class.tif")),
                    str(session.path("prediction_confidence.tif")),
                    str(session.path("ndvi.tif")),
                    str(session.path("final_class.tif")),
                    ndvi_conf_threshold
                )

                colorize_class_map(
                    str(session.path("final_class.tif")),
                    str(session.path("class_color_map.json")),
                    str(session.path("final_class_color.tif"))
                )

            st.session_state["inference_done"] = True
            set_progress_callback(None)

        # ---------------- DOWNLOAD ----------------
        if st.session_state["inference_done"]:
            session = st.session_state["session"]

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for f in ("final_class.tif", "final_class_color.tif",
                          "prediction_confidence.tif", "class_color_map.json"):
                    zf.write(session.path(f), arcname=f)

            zip_buf.seek(0)

            if st.download_button("📦 Download Results", zip_buf.getvalue(),
                                  file_name="gis_outputs.zip"):

                validate_email_config(email_cfg)
                csv_path = st.session_state["tracker"].export_csv(
                    session.path("run_report.csv")
                )

                send_csv_email(csv_path, **email_cfg)
                session.cleanup(force=True)
                st.session_state["cleanup_done"] = True
                st.success("Delivered & cleaned")

    # =========================================================
    # DOCS
    # =========================================================
    with tab_docs:
        ui_2_docs.render()


if __name__ == "__main__":
    main()

