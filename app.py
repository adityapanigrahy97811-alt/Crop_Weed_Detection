import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌿",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    font-weight: 700;
}
.block-container {
    padding-top: 2rem;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL (Cached)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train/weights/best.pt")

model = load_model()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("🌿 AgriVision AI")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Batch Processing", "About"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 0.9, 0.3, 0.05
)

st.sidebar.markdown("---")
st.sidebar.write("YOLOv8 + Streamlit")
st.sidebar.write("Precision Agriculture Analytics")

# ===================================================
# DASHBOARD PAGE
# ===================================================
if page == "Dashboard":

    st.title("🌱 Smart Crop & Weed Detection")
    st.caption("AI-powered selective pesticide analytics")

    uploaded_file = st.file_uploader(
        "Upload Field Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        with st.spinner("Running Detection..."):
            results = model.predict(
                image,
                conf=confidence_threshold,
                imgsz=320,
                verbose=False
            )

        annotated = results[0].plot()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(annotated, width="stretch")

        boxes = results[0].boxes
        data = []

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]
            data.append([label, round(confidence, 2)])

        df = pd.DataFrame(data, columns=["Class", "Confidence"])

        with col2:
            if not df.empty:
                weed_count = df[df["Class"] == "weed"].shape[0]
                crop_count = df[df["Class"] == "crop"].shape[0]

                st.metric("🌿 Weed Count", weed_count)
                st.metric("🌾 Crop Count", crop_count)
                st.metric("📊 Total Detections", len(df))
            else:
                st.warning("No objects detected.")

        if not df.empty:
            st.markdown("### 📊 Class Distribution")

            fig, ax = plt.subplots()
            df["Class"].value_counts().plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")
            st.pyplot(fig)

            st.markdown("### 📋 Detection Data")
            st.dataframe(df, width="stretch")

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Report",
                csv,
                "detection_report.csv",
                "text/csv"
            )

# ===================================================
# BATCH PROCESSING PAGE
# ===================================================
elif page == "Batch Processing":

    st.title("📂 Bulk Image Processing")
    st.caption("Efficient large-scale dataset analysis")

    uploaded_files = st.file_uploader(
        "Upload Multiple Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        progress_bar = st.progress(0)

        image_list = []
        file_names = []

        for file in uploaded_files:
            image_list.append(Image.open(file))
            file_names.append(file.name)

        results = model.predict(
            image_list,
            conf=confidence_threshold,
            imgsz=320,
            verbose=False
        )

        all_data = []

        for idx, result in enumerate(results):
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = model.names[class_id]

                all_data.append([
                    file_names[idx],
                    label,
                    round(confidence, 2)
                ])

            progress_bar.progress((idx + 1) / len(results))

        batch_df = pd.DataFrame(
            all_data,
            columns=["Image", "Class", "Confidence"]
        )

        if not batch_df.empty:

            total_weeds = batch_df[batch_df["Class"] == "weed"].shape[0]
            total_crops = batch_df[batch_df["Class"] == "crop"].shape[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("🌿 Total Weeds", total_weeds)
            col2.metric("🌾 Total Crops", total_crops)
            col3.metric("🖼 Images Processed", len(uploaded_files))

            st.markdown("### 📊 Batch Distribution")

            fig, ax = plt.subplots()
            batch_df["Class"].value_counts().plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")
            st.pyplot(fig)

            st.markdown("### 📋 Full Report")
            st.dataframe(batch_df, width="stretch")

            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Batch Report",
                csv,
                "batch_report.csv",
                "text/csv"
            )
        else:
            st.warning("No detections found.")

# ===================================================
# ABOUT PAGE
# ===================================================
elif page == "About":

    st.title("About AgriVision AI")

    st.markdown("""
    **AgriVision AI** is an AI-powered agricultural analytics platform designed 
    to detect weeds and crops using computer vision.

    ### 🔍 Core Capabilities
    - YOLOv8 object detection
    - Real-time inference
    - Bulk image processing
    - Detection analytics dashboard
    - CSV export functionality
    - Adjustable confidence threshold

    ### 🎯 Business Impact
    - Enables selective pesticide spraying
    - Reduces chemical waste
    - Minimizes crop contamination
    - Supports precision agriculture

    Built using Python, YOLOv8, and Streamlit.
    """)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("""
---
<center>
AgriVision AI © 2026 | Precision Agriculture Analytics Platform
</center>
""", unsafe_allow_html=True)
