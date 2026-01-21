# ============================================================
# ui_2_docs.py
# ============================================================
# PURPOSE:
# - Documentation & usage guide for GIS Classification App
# - Explain methodology, workflow, and outputs
# - Display author authority, ownership, and reach-out info
#
# NO TRAINING
# NO INFERENCE
# NO SIDE EFFECTS
# ============================================================

import streamlit as st


def render():
    # --------------------------------------------------------
    # PAGE TITLE
    # --------------------------------------------------------
    st.title("📘 Documentation & About")

    st.markdown(
        """
        This page explains **how to use the system**,  
        **what happens internally**, and **important constraints**.

        It is written for **GIS / Remote Sensing users**
        who want clarity, not black-box behavior.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # HOW TO USE
    # --------------------------------------------------------
    st.header("🧭 How to Use This Application")

    st.markdown(
        """
        **Step-by-step workflow:**

        1️⃣ Upload a **stacked satellite image** (GeoTIFF)  
        2️⃣ Upload the corresponding **NDVI image**  
        3️⃣ Define the number of classes you want to train  
        4️⃣ For each class:
            - Upload shapefile **ZIP**
            - Provide a **class label**
            - Choose a **color**
        5️⃣ Train the models  
        6️⃣ Run inference on the stack image  
        7️⃣ Download classified outputs for ArcGIS / QGIS

        ⚠️ **Bare land does NOT require a shapefile**  
        It is automatically derived using NDVI.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # METHODOLOGY
    # --------------------------------------------------------
    st.header("🧠 Methodology (What Happens Internally)")

    st.markdown(
        """
        This system follows a **GIS + Machine Learning workflow**:

        ### 🔹 Feature Construction
        - Multispectral bands from the stack image
        - NDVI appended as an explicit feature

        ### 🔹 Bare Land Logic
        - Derived automatically using NDVI range **0.0 – 0.2**
        - Assigned as **Class 0**
        - Sampling is capped to avoid class dominance

        ### 🔹 Models Used
        - **Random Forest** → robust, noise-tolerant
        - **XGBoost** → high-capacity learner
        - **Logistic Regression** → probability stabilizer

        ### 🔹 Ensemble Strategy
        - Probabilities from all models are combined
        - Reduces overfitting and improves stability

        ### 🔹 Post-processing
        - NDVI-based correction applied
        - Only affects **low-confidence pixels**
        """
    )

    st.divider()

    # --------------------------------------------------------
    # RETRAINING NOTE (IMPORTANT)
    # --------------------------------------------------------
    st.header("🔁 Training & Retraining Behavior")

    st.markdown(
        """
        - Models are **trained using your uploaded data**
        - There is **no global or pre-trained model**
        - New imagery or regions may require retraining

        This design keeps the system **simple, transparent, and region-adaptive**.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------
    st.header("📤 Outputs Explained")

    st.markdown(
        """
        The system produces **GIS-ready outputs**:

        - **Classified raster (GeoTIFF)**
          - Each pixel stores a class ID

        - **Colorized raster (RGB GeoTIFF)**
          - Uses colors chosen in the UI
          - Directly usable in ArcGIS / QGIS

        - **Confidence map**
          - Per-pixel prediction confidence

        All outputs preserve:
        - CRS
        - resolution
        - spatial alignment
        """
    )

    st.divider()

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------
    st.header("⚠️ Important Notes & Limitations")

    st.markdown(
        """
        - This is a **pixel-based classifier**
          - It does NOT generate polygons or boundaries

        - Shapefile attributes are ignored
          - Geometry only is used

        - Output quality depends on:
          - image quality
          - shapefile accuracy
          - NDVI correctness

        - On free servers:
          - training may take time
          - models are not persisted across restarts
        """
    )

    st.divider()

    # --------------------------------------------------------
    # AUTHOR & OWNERSHIP
    # --------------------------------------------------------
    st.header("👤 Author & Ownership")

    st.markdown(
        """
        **Abhinandan Sood**  
        *GIS • Remote Sensing • Machine Learning*

        🔗 https://www.linkedin.com/in/abhinandan-sood-098514242

        ---
        **Ownership & Intent**

        This application and workflow represent **original work**
        developed for professional GIS and remote sensing use cases.

        You are free to:
        - use it
        - test it
        - learn from it

        Please **credit the author** when sharing or demonstrating.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # CONTACT
    # --------------------------------------------------------
    st.header("📬 Reach Out — soodji.09@gmail.com")

    st.markdown(
        """
        For:
        - collaboration
        - custom GIS solutions
        - ML for remote sensing
        - research or industry projects

        Feel free to connect via LinkedIn or email.
        """
    )
