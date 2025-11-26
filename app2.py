"""
Pothole Detection System - Streamlit Web Application
MSc AI and Robotics Project - University of Hertfordshire
Student: Mahamad Alfaiz J. Saiyad (23022436)
Supervisor: Mohammad Rafea

Implementation Notes:
This application implements a web-based pothole detection interface using the
Streamlit framework and Ultralytics YOLO library. The implementation follows
standard patterns documented in:

1. Ultralytics YOLO Documentation (2023) - Model loading and inference
   Available at: https://docs.ultralytics.com/
   
2. Streamlit Documentation (2023) - Web interface components
   Available at: https://docs.streamlit.io/
   
3. OpenCV Documentation (2023) - Image processing operations
   Available at: https://docs.opencv.org/

Core logic adapted from Ultralytics YOLO inference examples with custom
modifications for pothole-specific deployment including CSV logging and
dual-mode operation (static/video).

Dependencies:
- streamlit==1.28.0
- ultralytics==8.0.196
- opencv-python==4.8.1
- Pillow==10.0.1
- numpy==1.24.3
- pandas==2.0.3
"""
import pandas as pd
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2

MODEL_PATHS = {
    "YOLOv5n": "./models/best.pt",
    "YOLOv8n": "./models/v8.pt",
    "YOLOv11n": "./models/best.pt"
}

SELECTED_MODEL = MODEL_PATHS["YOLOv5n"]
detection_model = YOLO(SELECTED_MODEL)

st.set_page_config(page_title="Pothole Detection", page_icon="🚧", layout="wide")

st.title("🚧 Automated Pothole Detection System")
st.markdown("**MSc AI & Robotics Project | University of Hertfordshire**")
st.markdown("---")

detection_mode = st.sidebar.radio(
    "Select Detection Mode",
    ["Static Image Analysis", "Live Video Stream"],
    help="Choose between single image upload or real-time webcam detection"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Model Info:**
- Architecture: YOLOv8n
- mAP@0.5: 98.84%
- Precision: 97.67%
- Recall: 96.38%
""")

if detection_mode == "Static Image Analysis":
    st.header("📸 Image Upload Mode")
    
    uploaded_file = st.file_uploader(
        "Upload road image containing potholes",
        type=["jpg", "png", "jpeg"],
        help="Supported formats: JPG, PNG, JPEG"
    )
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(input_image, use_column_width=True)
        
        image_array = np.array(input_image)
        image_bgr = image_array[:, :, ::-1].copy()
        
        with st.spinner("Detecting potholes..."):
            detection_results = detection_model(image_bgr)
        
        pothole_list = []
        annotated_image = image_bgr.copy()
        
        for result in detection_results:
            bboxes = result.boxes.xyxy
            confidences = result.boxes.conf
            
            for bbox, conf in zip(bboxes, confidences):
                x1, y1, x2, y2 = map(int, bbox)
                conf_score = round(float(conf), 2)
                
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    annotated_image, 
                    f"{conf_score}", 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    2
                )
                
                pothole_list.append({
                    "Image": uploaded_file.name,
                    "Confidence": conf_score,
                    "X1": x1, "Y1": y1, "X2": x2, "Y2": y2
                })
        
        with col2:
            st.subheader("Detection Results")
            st.image(annotated_image, use_column_width=True)
        
        st.markdown("---")
        
        if pothole_list:
            st.success(f"✅ Detected {len(pothole_list)} pothole(s)")
            
            avg_conf = np.mean([p["Confidence"] for p in pothole_list])
            max_conf = max([p["Confidence"] for p in pothole_list])
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Total Detected", len(pothole_list))
            metric_col2.metric("Avg Confidence", f"{avg_conf:.2%}")
            metric_col3.metric("Max Confidence", f"{max_conf:.2%}")
            
            with st.expander("View Detection Details"):
                details_df = pd.DataFrame(pothole_list)
                st.dataframe(details_df)
            
            results_df = pd.DataFrame(pothole_list)
            results_df.to_csv("pothole_detections.csv", mode="a", header=False, index=False)
            st.info("💾 Results saved to pothole_detections.csv")
        else:
            st.warning("⚠️ No potholes detected in this image")

elif detection_mode == "Live Video Stream":
    st.header("📹 Webcam Detection Mode")
    
    st.info("Click 'Start Detection' to begin real-time pothole detection using your webcam")
    
    camera_source = st.radio(
        "Camera Source:",
        ["Webcam (Default)", "IP Camera"],
        horizontal=True
    )
    
    if camera_source == "IP Camera":
        ip_address = st.text_input("Enter IP Camera URL:", "http://192.168.1.100:8080/video")
        video_src = ip_address
    else:
        video_src = 0
    
    start_btn = st.button("🎥 Start Detection", type="primary")
    
    if start_btn:
        video_cap = cv2.VideoCapture(video_src)
        
        if not video_cap.isOpened():
            st.error("🚫 Cannot access camera. Check connection and permissions.")
            st.stop()
        
        st.success("✅ Camera connected successfully")
        
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        frame_count = 0
        detection_count = 0
        
        while video_cap.isOpened():
            success, frame = video_cap.read()
            
            if not success:
                st.warning("Failed to read frame")
                break
            
            frame_count += 1
            
            detect_results = detection_model(frame)
            
            for result in detect_results:
                boxes = result.boxes.xyxy
                confs = result.boxes.conf
                
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    conf_val = round(float(conf), 2)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, 
                        f"{conf_val}", 
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        2
                    )
                    detection_count += 1
            
            cv2.putText(
                frame, 
                f"Frame: {frame_count} | Detections: {detection_count}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)
            
            stats_col1, stats_col2 = stats_placeholder.columns(2)
            stats_col1.metric("Frames Processed", frame_count)
            stats_col2.metric("Total Detections", detection_count)
        
        video_cap.release()
        st.success("✅ Detection session ended")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Pothole Detection System | MSc Project 2025 | University of Hertfordshire</small>
</div>
""", unsafe_allow_html=True)