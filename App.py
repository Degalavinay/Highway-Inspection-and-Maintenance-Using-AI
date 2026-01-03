# File: yolo_video_detection_live.py
# Run with: streamlit run yolo_video_detection_live.py
# Displays live detection of vehicles in video after upload, fixed for Windows

import streamlit as st
import cv2
from ultralytics import YOLO
import os
import tempfile
from pathlib import Path
import torch
import time
import numpy as np

# ==================== CONFIGURATION ====================
@st.cache_resource
def load_model(model_path):
    """Load the YOLO model."""
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Update the path.")
        return None
    model = YOLO(model_path)
    return model

# Update to your best.pt location (Windows path)
MODEL_PATH = "best.pt"
CLASS_NAMES = ['Auto', 'Bike', 'Bus', 'Car', 'Truck']
CONF_THRESH = 0.25
IOU_THRESH = 0.45

# ==================== STREAMLIT APP ====================
def main():
    st.title(" Vehicle Detection")
    st.write("Upload a video to see live vehicle detection (Auto, Bike, Bus, Car, Truck).")

    

    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader("Choose a test video", type=['mp4', 'avi', 'mov', 'mkv'])

    if uploaded_file is not None:
        # Save video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_path = tfile.name
        tfile.write(uploaded_file.read())
        tfile.close()  # Close file handle immediately

        st.success(f"Video uploaded: {uploaded_file.name}")

        # Video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Failed to open video file. Try another video.")
            try:
                os.unlink(video_path)
            except PermissionError:
                st.warning("Could not delete temp file due to access; will retry later.")
            st.stop()

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            st.write(f"Video stats: {total_frames} frames, {fps} FPS, {width}x{height}")
        else:
            st.warning("Could not determine total frames. Progress bar may be inaccurate.")
            total_frames = 1  # Avoid division by zero

        # Output path
        output_path = video_path.replace('.mp4', '_detected.mp4')

        # Placeholder for live display
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        # Process video
        if st.button("Start Detection"):
            with st.spinner("Starting detection..."):
                # Setup video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                frame_count = 0
                detections_per_frame = []
                class_counts = {cls: 0 for cls in CLASS_NAMES}

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Run YOLO prediction
                    results = model.predict(frame, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)

                    # Draw boxes
                    annotated_frame = results[0].plot()

                    # Convert for Streamlit display
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    # Display frame live
                    frame_placeholder.image(annotated_frame_rgb, caption=f"Frame {frame_count+1}/{total_frames}", use_column_width=True)

                    # Count detections
                    if results[0].boxes is not None:
                        num_dets = len(results[0].boxes)
                        detections_per_frame.append(num_dets)
                        classes = results[0].boxes.cls.cpu().numpy()
                        for cls in classes:
                            class_counts[CLASS_NAMES[int(cls)]] += 1

                    # Write frame to output
                    out.write(annotated_frame)

                    # Update progress
                    frame_count += 1
                    status_placeholder.write(f"Processing frame {frame_count}/{total_frames} | Detections: {num_dets if 'num_dets' in locals() else 0}")
                    st.progress(frame_count / total_frames)

                    # Small delay to simulate real-time (adjust for smoother display)
                    time.sleep(0.01)

                # Cleanup
                cap.release()
                out.release()
                cv2.destroyAllWindows()

                # Final stats
                avg_dets = sum(detections_per_frame) / len(detections_per_frame) if detections_per_frame else 0
                st.success(f"Live detection complete! Average detections per frame: {avg_dets:.2f}")
                st.write("Class detection counts:")
                for cls, count in class_counts.items():
                    st.write(f"  {cls}: {count}")

                # Download option
                with open(output_path, 'rb') as f:
                    st.download_button("⬇️ Download Detected Video", f, file_name=f"{uploaded_file.name}_detected.mp4")

                # Final frame preview
                cap = cv2.VideoCapture(output_path)
                ret, preview = cap.read()
                if ret:
                    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                    st.image(preview_rgb, caption="Final Sample Detected Frame", use_column_width=True)
                cap.release()

            # Cleanup temp file with retry
            for _ in range(5):
                try:
                    os.unlink(video_path)
                    st.write(f"✅ Temporary file {video_path} deleted.")
                    break
                except PermissionError:
                    st.warning("PermissionError: File in use. Retrying in 1 second...")
                    time.sleep(1)
                except Exception as e:
                    st.error(f"Failed to delete temp file: {str(e)}")
                    break

if __name__ == "__main__":
    main()