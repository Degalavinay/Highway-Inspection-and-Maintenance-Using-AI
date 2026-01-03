
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import os
import base64
from collections import Counter
import pandas as pd

# Set page config
st.set_page_config(page_title="Highway Inspection AI", layout="wide")

# === Logo and Styled Header ===
logo_path = "AiSPRY logo.jpg"

# Read and encode image
encoded_logo = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_bytes = f.read()
        encoded_logo = base64.b64encode(logo_bytes).decode()

# Custom HTML and CSS
st.markdown(f"""
    <style>
        .main-title {{
            text-align: center;
            font-size: 40px;
            font-weight: 700;
            color: #ff6600;
            margin-top: 20px;
        }}
        .sub-text {{
            text-align: center;
            font-size: 18px;
            color: #555;
            margin-bottom: 30px;
        }}
        .footer {{
            text-align: center;
            color: gray;
            font-size: 14px;
            padding: 20px 0;
            border-top: 1px solid #e6e6e6;
            margin-top: 50px;
        }}
    </style>

    <div style="text-align: center;">
        {'<img src="data:image/jpeg;base64,' + encoded_logo + '" width="250" style="margin-bottom:10px;">' if encoded_logo else ''}
    </div>
    <div class="main-title">Highway Inspection with AI</div>
    <div class="sub-text">Stay SPRY. Stay Smart.</div>
""", unsafe_allow_html=True)

# === Load YOLO model ===
model_path = "yolov8m_finetuned.pt"
try:
    model = YOLO(model_path)
    st.success(f"✅ Model loaded successfully: {model_path}")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# === Image Upload ===
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        CONF_THRESHOLD = 0.25
        IOU_THRESHOLD = 0.4

        # Predict
        results = model.predict(img_bgr, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        result = results[0]
        boxes = result.boxes
        class_names = model.names

        # Filter predictions
        filtered_detections = []
        for box in boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            if conf >= CONF_THRESHOLD:
                filtered_detections.append({
                    "Class": class_names[class_id],
                    "Confidence": f"{conf:.2f}"
                })

        # Annotated image
        annotated_img = result.plot(conf=CONF_THRESHOLD)

        # Tabs for output
        tab1, tab2, tab3 = st.tabs(["🖼️ Annotated Image", "📊 Summary", "🧪 Raw Debug"])

        with tab1:
            st.image(annotated_img, caption=f"Detections (Conf ≥ {CONF_THRESHOLD})", use_column_width=True)
            buf = io.BytesIO()
            Image.fromarray(annotated_img).save(buf, format="PNG")
            st.download_button("📥 Download Image", data=buf.getvalue(), file_name="clean_detections.png", mime="image/png")

        with tab2:
            if filtered_detections:
                st.subheader("Detected Objects")
                df = pd.DataFrame(filtered_detections)
                st.dataframe(df, use_container_width=True)
                st.subheader("Class Count Summary")
                counts = pd.DataFrame(Counter([d["Class"] for d in filtered_detections]).items(), columns=["Class", "Count"])
                st.table(counts)
            else:
                st.info("No objects detected above the confidence threshold.")

        with tab3:
            st.subheader("🔍 All Detections (Raw Boxes)")
            raw_output = [
                f"{class_names[int(box.cls)]} - Conf: {float(box.conf):.2f}" for box in boxes
            ]
            st.write(raw_output or "No detections found.")

    except Exception as e:
        st.error(f"Error processing image: {e}")

# === Footer ===
st.markdown('<div class="footer">© 2025 Vinay | YOLOv8 Highway AI | Powered by AiSPRY</div>', unsafe_allow_html=True)
