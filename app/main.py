import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# --- CONFIGURATION BASED ON YOUR TRAINING SCRIPT ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'scripts', 'best_model.h5')
IMAGE_SIZE = (150, 150)                 # Matches your script's target_size

# Update this list to match the exact alphabetical order of your dataset folders
# Example common classes: ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# --- PAGE SETUP ---
st.set_page_config(page_title="Brain Tumor Detection UI", page_icon="🧠", layout="centered")

@st.cache_resource
def load_my_model():
    """Loads the compiled neural network model once into memory."""
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        return None

model = load_my_model()

# --- USER INTERFACE ---
st.title("🧠 Brain Tumor Diagnostic Interface")
st.write("Upload a patient's brain MRI scan below to perform categorical classification analysis.")

if model is None:
    st.error(f"⚠️ Saved model file not found at path: '{MODEL_PATH}'. Please place your trained .h5 or .keras file there or update the MODEL_PATH variable at the top of app.py.")
else:
    # File upload interface
    uploaded_file = st.file_uploader("Select an MRI image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and render image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan for Processing", use_container_width=True)        
        st.info("Running Deep Learning network inference...")
        
        # --- PREPROCESSING MATCHING DATA GENERATOR ---
        # 1. Resize to 150x150
        img_resized = image.resize(IMAGE_SIZE)
        
        # 2. Ensure 3 channels (RGB)
        if img_resized.mode != 'RGB':
            img_resized = img_resized.convert('RGB')
            
        # 3. Convert to numpy array and rescale (1./255)
        img_array = np.array(img_resized) / 255.0
        
        # 4. Expand dimensions for batch format: (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # --- PREDICTION ---
        predictions = model.predict(img_batch)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx] * 100
        
        st.markdown("---")
        st.subheader("Diagnostic Metrics")
        
        # Handle safety fallback if list length mismatches your custom classes
        if predicted_class_idx < len(CLASS_NAMES):
            detected_label = CLASS_NAMES[predicted_class_idx]
        else:
            detected_label = f"Class Index {predicted_class_idx}"

        # Style output based on whether a tumor type is detected
        if "No Tumor" in detected_label:
            st.success(f"✅ **Result:** {detected_label} (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"⚠️ **Result:** {detected_label} Detected (Confidence: {confidence:.2f}%)")
            
        # Optional: Expandable drop-down showing full confidence break-down
        with st.expander("View Full Softmax Probabilities"):
            for idx, prob in enumerate(predictions):
                label_text = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
                st.write(f"**{label_text}**: {prob * 100:.2f}%")