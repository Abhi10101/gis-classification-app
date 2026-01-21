# ============================================================
# ui_1_main.py
# ============================================================
# PURPOSE:
# - Main UI page for GIS Classification App
# - Collect user inputs
# - Display rules, constraints, and author authority
#
# NO TRAINING
# NO INFERENCE
# NO FILE SYSTEM SIDE EFFECTS
# ============================================================

import streamlit as st


def render():
    # --------------------------------------------------------
    # PAGE TITLE
    # --------------------------------------------------------
    st.title("🛰️ GIS Image Classification System")

    st.markdown(
        """
        A **GIS-focused image classification tool** designed for  
        **remote sensing, agriculture, and land-use analysis**.

        This system allows you to:
        - Train your own classification model
        - Use NDVI-based logic for bare land detection
        - Generate ArcGIS-ready classified outputs
        """
    )

    st.divider()

    # --------------------------------------------------------
    # AUTHOR / AUTHORITY
    # --------------------------------------------------------
    st.markdown(
        """
        #### 👤 Author & Authority

        **Abhinandan Sood**  
        *GIS • Remote Sensing • Machine Learning*

        🔗 https://www.linkedin.com/in/abhinandan-sood-098514242
        """
    )

    st.divider()

    # --------------------------------------------------------
    # RULES
    # --------------------------------------------------------
    st.subheader("📌 Important Rules")

    st.warning(
        """
        • **Bare land (Class 0)** is derived automatically using NDVI (0.0–0.2)
        • **Do NOT upload shapefile for bare land**
        • Shapefiles are used ONLY for geometry
        • Stack image and NDVI image must be aligned
        • Shapefiles must be uploaded as ZIP files
        • Pixel-based classification (no polygon extraction)
        """
    )

    st.divider()

    # --------------------------------------------------------
    # INPUT FILES
    # --------------------------------------------------------
    st.subheader("📂 Input Data")

    stack_file = st.file_uploader(
        "Upload Stack Image (GeoTIFF)",
        type=["tif", "tiff"]
    )

    ndvi_file = st.file_uploader(
        "Upload NDVI Image (GeoTIFF)",
        type=["tif", "tiff"]
    )

    if stack_file and not ndvi_file:
        st.warning("NDVI image is required with stack image")

    if ndvi_file and not stack_file:
        st.warning("Stack image is required with NDVI image")

    st.divider()

    # --------------------------------------------------------
    # USER CLASSES
    # --------------------------------------------------------
    st.subheader("🏷️ Define Classes (Except Bare Land)")

    num_classes = st.number_input(
        "Number of classes you want to define",
        min_value=1,
        max_value=20,
        step=1
    )

    class_inputs = []
    valid_class_count = 0

    for i in range(int(num_classes)):
        st.markdown(f"### Class {i + 1}")

        col1, col2 = st.columns(2)

        with col1:
            class_label = st.text_input(
                f"Class Label {i + 1}",
                placeholder="e.g. crop, building, forest"
            )

        with col2:
            class_color = st.color_picker(
                f"Class Color {i + 1}",
                value="#00B400"
            )

        shapefile_zip = st.file_uploader(
            f"Upload Shapefile ZIP for Class {i + 1}",
            type=["zip"],
            key=f"shp_{i}"
        )

        # FIX: basic consistency checks
        if shapefile_zip and not class_label:
            st.error("Class label is required when uploading shapefile")

        if class_label and shapefile_zip:
            valid_class_count += 1

        class_inputs.append({
            "label": class_label,
            "color": class_color,
            "zip": shapefile_zip
        })

        st.divider()

    if num_classes > 0 and valid_class_count == 0:
        st.warning("At least one valid class (label + shapefile) is required")

    # --------------------------------------------------------
    # RETURN INPUTS ONLY
    # --------------------------------------------------------
    return {
        "stack_file": stack_file,
        "ndvi_file": ndvi_file,
        "classes": class_inputs
    }
