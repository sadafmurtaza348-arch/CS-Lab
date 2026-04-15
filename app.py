import os
# Linux servers par graphical errors se bachne ke liye ye environment variable set kiya hai
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="AI Vision Detector", layout="centered")

# --- UI Header ---
st.title("🔍 AI Object Detector")
st.write("Apni image upload krein aur AI automatically objects detect kr k dikhaye ga.")
st.markdown("---")

# --- Load Pre-trained Model ---
# Pehli dafa run krny pr model download hoga, is liye thora waqt lag sakta hai
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolov8n.pt')  # Nano version: Lightweight aur fast
        return model
    except Exception as e:
        st.error(f"Model load krny mein masla hua: {e}")
        return None

model = load_model()

# --- Image Upload Section ---
uploaded_file = st.file_uploader("Image select krein...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Open image
    image = Image.open(uploaded_file)
    
    # UI Columns for Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Original Image")
        st.image(image, use_container_width=True)

    # Perform Detection
    with st.spinner('AI analysis kr rha hai...'):
        # Model inference
        results = model(image)
        
        # Result image (with boxes)
        res_plotted = results[0].plot()
        # Convert BGR (OpenCV format) to RGB (PIL/Streamlit format)
        res_image = Image.fromarray(res_plotted[:, :, ::-1]) 

    with col2:
        st.success("AI Detection")
        st.image(res_image, use_container_width=True)

    # --- Details Section ---
    st.markdown("---")
    st.subheader("📊 Detection Summary")
    
    # Get detected objects count
    names = model.names
    detected_objects = results[0].boxes.cls.tolist()
    
    if len(detected_objects) > 0:
        counts = {}
        for obj_id in detected_objects:
            obj_name = names[int(obj_id)]
            counts[obj_name] = counts.get(obj_name, 0) + 1
        
        # Displaying results in a nice format
        for item, count in counts.items():
            st.write(f"✅ Found **{count} {item.capitalize()}**")
    else:
        st.write("Koi object detect nahi hua.")

elif model is None:
    st.error("AI model load nahi ho saka. Baraye meherbani logs check krein.")
else:
    st.warning("Shuru krny k liye koi image upload krein.")
