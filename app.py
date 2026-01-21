# ============================================================
# app.py
# ============================================================
# Production-safe Streamlit app
# Fully wired with:
# - session isolation
# - training
# - inference
# - NDVI rules
# - colorization
# - CSV MLOps report
# - Gmail delivery
# - guaranteed cleanup
# ============================================================
import streamlit as st

st.set_page_config(
    page_title="GIS Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

import streamlit as st
from pathlib import Path
import json
import zipfile
import io
import gc
import yaml
import atexit

import ui_1_main
import ui_2_docs

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
    from app_validators import validate_inputs
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
# LIMITS (HARD SAFETY)
# ------------------------------------------------------------
MAX_RASTER_MB = 1500
MAX_CLASS_ZIPS = 10
MAX_ZIP_MB = 50


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
def load_config() -> dict:
    cfg = Path("config.yaml")
    if not cfg.exists():
        return {}
    with open(cfg, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_email_config(cfg: dict):
    for k in ("sender", "smtp_server", "smtp_port", "app_password"):
        if not cfg.get(k):
            raise ValueError(f"Missing email config key: {k}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    st.set_page_config(
        page_title="GIS Image Classification | Abhinandan Sood",
        layout="wide"
    )

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------
    config = load_config()
    inference_cfg = config.get("inference", {})
    ensemble_cfg = config.get("ensemble", {})
    ndvi_cfg = config.get("ndvi_rules", {})
    email_cfg = config.get("email", {})

    tile_size = int(inference_cfg.get("tile_size", 512))
    ensemble_weights = ensemble_cfg.get("weights", None)
    ndvi_conf_threshold = ndvi_cfg.get("confidence_threshold", None)

    # --------------------------------------------------------
    # SESSION STATE INIT (NO KEYERROR POSSIBLE)
    # --------------------------------------------------------
    defaults = {
        "session": None,
        "tracker": None,
        "models_trained": False,
        "inference_done": False,
        "cleanup_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # --------------------------------------------------------
    # FAILSAFE CLEANUP (PROCESS EXIT)
    # --------------------------------------------------------
    def _final_cleanup():
        sess = st.session_state.get("session")
        if sess and not st.session_state.get("cleanup_done"):
            try:
                sess.cleanup(force=True)
            except Exception:
                pass

    atexit.register(_final_cleanup)

    # --------------------------------------------------------
    # UI TABS
    # --------------------------------------------------------
    tab_home, tab_docs = st.tabs(["🏠 Home", "📘 Documentation"])

    # ========================================================
    # HOME
    # ========================================================
    with tab_home:

        inputs = ui_1_main.render()

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------
        st.divider()
        st.subheader("🧠 Model Training")

        if st.button("🚀 Train Models"):

            stack_file = inputs.get("stack_file")
            ndvi_file = inputs.get("ndvi_file")
            classes = inputs.get("classes", [])

            # ---------------- VALIDATION ----------------
            if not stack_file or not ndvi_file:
                st.error("Stack and NDVI rasters are required")
                st.stop()

            if stack_file.size / 1e6 > MAX_RASTER_MB:
                st.error("Stack raster too large")
                st.stop()

            if ndvi_file.size / 1e6 > MAX_RASTER_MB:
                st.error("NDVI raster too large")
                st.stop()

            if len(classes) > MAX_CLASS_ZIPS:
                st.error("Too many class shapefiles")
                st.stop()

            for cls in classes:
                if cls.get("zip") and cls["zip"].size / 1e6 > MAX_ZIP_MB:
                    st.error("One or more class ZIPs too large")
                    st.stop()

            if validate_inputs:
                ok, msg = validate_inputs(inputs)
                if not ok:
                    st.error(msg)
                    st.stop()

            # ---------------- SESSION ----------------
            session = SessionManager()
            out_dir = session.create()
            init_logger(out_dir)

            tracker = RunTracker(session_id=out_dir.name)
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

                    # Save rasters
                    stack_path = session.path("stack.tif")
                    ndvi_path = session.path("ndvi.tif")
                    stack_path.write_bytes(stack_file.read())
                    ndvi_path.write_bytes(ndvi_file.read())

                    # Prepare class info
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

                    with open(session.path("class_color_map.json"), "w", encoding="utf-8") as f:
                        json.dump(color_map, f, indent=2)

                    build_dataset(str(stack_path), str(ndvi_path), shapefile_info, str(out_dir))
                    progress.progress(30)

                    prepare_features(
                        str(session.path("X.npy")),
                        str(session.path("y.npy")),
                        str(out_dir)
                    )
                    progress.progress(50)

                    train_random_forest(str(session.path("X_clean.npy")),
                                         str(session.path("y_clean.npy")),
                                         str(out_dir))
                    progress.progress(70)

                    train_xgboost(str(session.path("X_clean.npy")),
                                  str(session.path("y_clean.npy")),
                                  str(out_dir))
                    progress.progress(85)

                    train_logistic_regression(str(session.path("X_clean.npy")),
                                              str(session.path("y_clean.npy")),
                                              str(out_dir))
                    progress.progress(100)

                st.session_state["models_trained"] = True
                st.success("Training completed successfully")

            except Exception as e:
                log_event(f"Training failed: {e}", "ERROR")
                session.cleanup(force=True)
                st.error("Training failed")
                st.stop()

            finally:
                gc.collect()

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------
        st.divider()
        st.subheader("🛰️ Inference")

        if st.button("▶️ Run Inference", disabled=not st.session_state["models_trained"]):

            session = st.session_state["session"]
            tracker = st.session_state["tracker"]

            try:
                with st.status("Running inference...", expanded=True):
                    progress = st.progress(0)
                    set_progress_callback(lambda p, m: progress.progress(p, m))

                    run_inference(
                        stack_path=str(session.path("stack.tif")),
                        ndvi_path=str(session.path("ndvi.tif")),
                        model_dir=str(session.path()),
                        out_dir=str(session.path()),
                        tile_size=tile_size,
                        ensemble_weights=ensemble_weights
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
                st.success("Inference completed")

            except Exception as e:
                log_event(f"Inference failed: {e}", "ERROR")
                session.cleanup(force=True)
                st.error("Inference failed")
                st.stop()

            finally:
                set_progress_callback(None)
                gc.collect()

        # ----------------------------------------------------
        # DOWNLOAD + EMAIL + CLEANUP
        # ----------------------------------------------------
        if st.session_state["inference_done"]:

            session = st.session_state["session"]
            tracker = st.session_state["tracker"]

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in (
                    "final_class.tif",
                    "final_class_color.tif",
                    "prediction_confidence.tif",
                    "class_color_map.json",
                ):
                    zf.write(session.path(f), arcname=f)

            zip_buf.seek(0)

            if st.download_button("📦 Download Results", zip_buf.getvalue(),
                                  file_name="gis_outputs.zip"):

                try:
                    validate_email_config(email_cfg)
                    csv_path = tracker.export_csv(session.path("run_report.csv"))

                    send_csv_email(
                        csv_path=csv_path,
                        sender_email=email_cfg["sender"],
                        receiver_email="soodji57@gmail.com",
                        smtp_server=email_cfg["smtp_server"],
                        smtp_port=email_cfg["smtp_port"],
                        app_password=email_cfg["app_password"]
                    )

                except Exception as e:
                    log_event(f"Email failed: {e}", "ERROR")

                finally:
                    session.cleanup(force=True)
                    st.session_state["cleanup_done"] = True
                    st.success("Results delivered and cleaned up")

    # ========================================================
    # DOCS
    # ========================================================
    with tab_docs:
        ui_2_docs.render()


if __name__ == "__main__":
    main()

