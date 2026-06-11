
import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageOps

# --- 1. PAGE LAYOUT CONFIGURATION ---
st.set_page_config(
    page_title="Ophthalmic AI - Cataract Screening",
    page_icon="👁️",
    layout="centered"
)

# --- 2. CACHED ASSET LOADER ---
@st.cache_resource
def load_prediction_assets():
    # Construct the path to your 06 model file
    model_path = os.path.join("saved_models", "06_cataract_cnn_model.keras")

    # Load the compiled Keras CNN model matrix
    model = tf.keras.models.load_model(model_path)

    # Ensure the metadata attribute is explicitly present
    if not hasattr(model, "class_indices_map"):
        model.class_indices_map = {"0": "Cataract", "1": "Normal"}

    labels_mapping = model.class_indices_map
    return model, labels_mapping

# Initialize the assets
try:
    model, labels_mapping = load_prediction_assets()
except Exception as e:
    st.error("Error loading model artifacts. Please verify that 'saved_models/06_cataract_cnn_model.keras' exists.")
    st.stop()

# --- 3. USER INTERFACE UI ---
st.title("👁️ Diagnostic Cataract Screening Tool")
st.write("An Explainable AI prototype for automated cataract binary classification from grayscale ocular imaging.")
st.markdown("---")

st.sidebar.header("System Parameters")
st.sidebar.info(
    "Upload a standard ocular image (JPEG/PNG). The system crops and downscales "
    "the matrix to 64x64 pixels in grayscale to match the input layer configurations "
    "of the convolutional neural network framework."
)

# File Uploader Widget
uploaded_file = st.file_uploader("Upload Ocular Clinical Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)

    # Split the screen into two columns for clear presentation
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Clinical Image", use_container_width=True)

    # --- 4. IMAGE PREPROCESSING PIPELINE ---
    with st.spinner("Processing image dimensions..."):
        # 1. Transform to Grayscale to match training color_mode='grayscale' (1 channel)
        gray_image = ImageOps.grayscale(image)
        # 2. Resize to 64x64 pixels to match the input shape layer
        resized_image = gray_image.resize((64, 64))
        # 3. Convert to a standard NumPy array and normalize intensities to [0, 1]
        img_array = np.array(resized_image) / 255.0
        # 4. Expand dimensions to create a batch format: shape becomes (1, 64, 64, 1)
        img_tensor = np.expand_dims(img_array, axis=(0, -1))

    # --- 5. MODEL INFERENCE ---
    with st.spinner("Running convolutional feature extraction..."):
        prediction_prob = model.predict(img_tensor)[0][0]

    # Standard threshold optimization logic
    if prediction_prob > 0.5:
        predicted_idx = "1"
        confidence = prediction_prob
    else:
        predicted_idx = "0"
        confidence = 1.0 - prediction_prob

    diagnosis = labels_mapping[predicted_idx]

    # --- 6. DISPLAY RESULTS ---
    with col2:
        st.subheader("Diagnostic Output")

        # Visually accentuating the clinical alert state
        if "cataract" in diagnosis.lower():
            st.error(f"Detected Classification: **{diagnosis.upper()}**")
        else:
            st.success(f"Detected Classification: **{diagnosis.upper()}**")

        st.metric(label="Inference Confidence", value=f"{confidence * 100:.2f}%")
        st.progress(float(confidence))

    st.markdown("---")
    st.caption("**Disclaimer:** This digital screening application is built strictly for academic verification purposes and does not replace professional diagnostic evaluations.")
