import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Page Config
st.set_page_config(page_title="AeroGuard | Drone vs Bird Classifier", page_icon="🛸", layout="centered")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMarkdown h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .status-badge {
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        background: #1e293b;
        border: 1px solid #334155;
        color: #10b981;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        text-align: center;
        margin-top: 2rem;
    }
    .drone-text {
        color: #ff4b4b;
        font-weight: 900;
        font-size: 2.5rem;
        text-shadow: 0 0 20px rgba(255, 75, 75, 0.3);
    }
    .bird-text {
        color: #28a745;
        font-weight: 900;
        font-size: 2.5rem;
        text-shadow: 0 0 20px rgba(40, 167, 69, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛡️ AeroGuard Control")
st.sidebar.markdown('<div class="status-badge">🟢 System Status: Active</div>', unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.info("AeroGuard uses AI to distinguish between birds and drones in real-time. Upload an image to begin.")

# Main UI
st.title("🛸 AeroGuard")
st.subheader("Aerial Intelligence Security Dashboard")

# Model Path (resolved relative to this script for Streamlit Cloud compatibility)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model.keras')
LOG_FILE = os.path.join(BASE_DIR, 'detection_log.csv')

def log_detection(result, confidence):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([[now, result, f"{confidence:.2%}"]], columns=["Timestamp", "Object Type", "Confidence"])
    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        log_df = pd.concat([log_df, new_entry], ignore_index=True)
    else:
        log_df = new_entry
    log_df.to_csv(LOG_FILE, index=False)

@st.cache_resource
def load_classifier():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_classifier()

if model is None:
    st.warning("⚠️ High-Performance Classifier not found. Please run the training script first.")
    st.code("python train_model.py")
else:
    uploaded_file = st.file_uploader("Upload Aerial Surveillance Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Preprocessing
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display image
        st.image(image, caption="Surveillance Feed", use_container_width=True)
        
        # Process for model
        img_array = image.resize((224, 224))
        img_array = np.array(img_array) / 255.0  # Normalized scaling
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing spectral signatures..."):
            prediction = model.predict(img_array, verbose=0)[0][0]
            result = "DRONE" if prediction > 0.5 else "BIRD"
            confidence = float(prediction) if prediction > 0.5 else 1.0 - float(prediction)
            log_detection(result, confidence)

        # UI Results
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        if prediction > 0.5:
            st.markdown('<p class="drone-text">🚨 DRONE DETECTED</p>', unsafe_allow_html=True)
            st.error(f"Confidence: {confidence:.2%}")
        else:
            st.markdown('<p class="bird-text">🐦 BIRD IDENTIFIED</p>', unsafe_allow_html=True)
            st.success(f"Confidence: {confidence:.2%}")
        
        st.progress(confidence)
        st.markdown('</div>', unsafe_allow_html=True)

        if os.path.exists(LOG_FILE):
            st.divider()
            st.subheader("📊 Recent Detection History")
            history_df = pd.read_csv(LOG_FILE).tail(5)
            st.table(history_df)
    else:
        st.info("Please upload a .jpg or .png image to analyze.")

st.divider()
st.caption("AeroGuard Project | Senior AI Architecture Deployment")
