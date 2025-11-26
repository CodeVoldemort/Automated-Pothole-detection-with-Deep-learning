"""
Pothole Detection System - Streamlit Web Application
MSc AI and Robotics Project - University of Hertfordshire
Student: Mahamad Alfaiz J. Saiyad (23022436)
Supervisor: Mohammad Rafea

Implementation Notes:
This version integrates the recommended 'streamlit-webrtc' component 
for secure and non-blocking live webcam detection.

Dependencies:
- streamlit>=1.28.0
- ultralytics>=8.0.196
- opencv-python>=4.8.1
- Pillow>=10.0.1
- numpy>=1.24.3
- pandas>=2.0.3
- streamlit-webrtc (NEW)
- av (NEW)
"""
import pandas as pd
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# --- 0. Configuration and Model Loading ---
MODEL_PATHS = {
    "YOLOv5n": "best.pt",
    "YOLOv8n": "v8.pt",
    "YOLOv11n": "v11.pt"
}

# NOTE: Ensure the path to your desired model is correct (e.g., v8.pt)
SELECTED_MODEL = MODEL_PATHS["YOLOv8n"] 
# Load the model outside of functions/loops for efficiency
try:
    detection_model = YOLO(SELECTED_MODEL)
except FileNotFoundError:
    st.error(f"Model file not found: {SELECTED_MODEL}. Please ensure the model file is in the same directory.")
    st.stop()


st.set_page_config(page_title="Pothole Detection", page_icon="🚧", layout="wide")

st.title("🚧 Automated Pothole Detection System")
st.markdown("**MSc AI & Robotics Project | University of Hertfordshire**")
st.markdown("---")

# --- 1. Sidebar and Mode Selection ---
detection_mode = st.sidebar.radio(
    "Select Detection Mode",
    ["Static Image Analysis", "Live Video Stream"],
    help="Choose between single image upload or real-time webcam detection"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Model Info:**
- Architecture: YOLOv8n (Simulated)
- mAP@0.5: 98.84%
- Precision: 97.67%
- Recall: 96.38%
""")

# --- 2. Static Image Analysis Mode ---
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
        
        # Convert PIL image to NumPy array (RGB) -> OpenCV BGR format for processing
        image_array = np.array(input_image)
        image_bgr = image_array[:, :, ::-1].copy() # RGB to BGR conversion
        
        with st.spinner("Detecting potholes..."):
            # Perform inference
            detection_results = detection_model(image_bgr)
        
        pothole_list = []
        annotated_image = image_bgr.copy()
        
        for result in detection_results:
            bboxes = result.boxes.xyxy
            confidences = result.boxes.conf
            
            for bbox, conf in zip(bboxes, confidences):
                x1, y1, x2, y2 = map(int, bbox)
                conf_score = round(float(conf), 2)
                
                # Draw bounding box and text using OpenCV
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3) # BGR: Green
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
        
        # Convert BGR annotated image back to RGB for Streamlit display
        annotated_image_rgb = annotated_image[:, :, ::-1]
        
        with col2:
            st.subheader("Detection Results")
            st.image(annotated_image_rgb, use_column_width=True, channels="RGB")
        
        st.markdown("---")
        
        if pothole_list:
            st.success(f"✅ Detected {len(pothole_list)} pothole(s)")
            
            # --- Metrics Display ---
            avg_conf = np.mean([p["Confidence"] for p in pothole_list])
            max_conf = max([p["Confidence"] for p in pothole_list])
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Total Detected", len(pothole_list))
            metric_col2.metric("Avg Confidence", f"{avg_conf:.2%}")
            metric_col3.metric("Max Confidence", f"{max_conf:.2%}")
            
            # --- Details Table and CSV Logging ---
            with st.expander("View Detection Details"):
                details_df = pd.DataFrame(pothole_list)
                st.dataframe(details_df)
            
            # Append results to CSV (creates file if not exists)
            results_df = pd.DataFrame(pothole_list)
            # Check if file exists to decide on header inclusion
            try:
                with open("pothole_detections.csv", 'r') as f:
                    header = False
            except FileNotFoundError:
                header = True
            
            results_df.to_csv("pothole_detections.csv", mode="a", header=header, index=False)
            st.info("💾 Results saved to pothole_detections.csv")
        else:
            st.warning("⚠️ No potholes detected in this image")

# --- 3. Live Video Stream Mode (WebRTC) ---
elif detection_mode == "Live Video Stream":
    st.header("📹 Live Webcam Detection Mode")
    
    st.info("Click **'Start'** to begin real-time pothole detection using your webcam.")
    
    # --- Video Processor Class Definition ---
    class PotholeDetectorProcessor(VideoProcessorBase):
        """
        Processes video frames using the YOLO model for real-time pothole detection.
        """
        def __init__(self, model):
            # Pass the loaded YOLO model to the processor instance
            self.model = model
            self.total_detections = 0
            self.frame_count = 0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # Convert the AV object frame to an OpenCV (numpy) BGR array
            img = frame.to_ndarray(format="bgr24")

            # Perform YOLO inference (set verbose=False to clean up console output)
            results = self.model(img, verbose=False) 

            frame_detections = 0
            
            # Draw bounding boxes
            for result in results:
                # Ensure boxes and confidences are available
                if result.boxes:
                    for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                        x1, y1, x2, y2 = map(int, box)
                        conf_score = round(float(conf), 2)

                        # Draw rectangle and confidence score
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # BGR: Green
                        cv2.putText(
                            img, 
                            f"Pothole: {conf_score}", 
                            (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0), 
                            2
                        )
                        frame_detections += 1
            
            # Update counters
            self.total_detections += frame_detections
            self.frame_count += 1
            
            # Display stats on the frame
            cv2.putText(
                img, 
                f"Frame: {self.frame_count} | Detections: {self.total_detections}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # Convert the modified image (numpy BGR array) back to an AV object
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # --- WebRTC Streamer Setup ---
    # Configuration for STUN server to help connection over the internet
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    ctx = webrtc_streamer(
        key="pothole-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: PotholeDetectorProcessor(detection_model), # Pass the model
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": "environment"} # <--- Explicitly request back camera
            }, 
            "audio": False
        },
        async_processing=True,
    )
    
    # --- Display Real-Time Stats ---
    stats_placeholder = st.empty()
    
    if ctx.state.playing:
        st.success("Live Detection Running...")
        
        # Continuously update the metrics
        while ctx.state.playing:
            processor = ctx.video_processor
            if processor:
                stats_col1, stats_col2 = stats_placeholder.columns(2)
                stats_col1.metric("Frames Processed", processor.frame_count)
                stats_col2.metric("Total Detections", processor.total_detections)
                # Small sleep to prevent constant re-rendering/high CPU usage
                import time
                time.sleep(0.1)
    
    elif ctx.state.stopped:
        st.info("Detection stream stopped.")


# --- 4. Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Pothole Detection System | MSc Project 2025 | University of Hertfordshire</small>
</div>

""", unsafe_allow_html=True)

