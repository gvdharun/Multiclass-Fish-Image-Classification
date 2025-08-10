# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image


# Load Trained Model
# Change path if your model is stored elsewhere
MODEL_PATH = "models/best_inceptionv3_finetuned.h5"  # or "models/best_custom_cnn.h5"
model = load_model(MODEL_PATH)

# List of class names in the same order as during training
# IMPORTANT: match this to train_gen.class_indices.keys()
class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

IMG_SIZE = (224, 224)  # same as training

# Streamlit UI
st.set_page_config(page_title="🐟 Fish Image Classifier", layout="centered")

st.title("🐟 Multiclass Fish Image Classification")
st.markdown("Upload a fish image to predict its species.")

# File uploader
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0  # rescale to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    # Prediction
    preds = model.predict(img_array)
    pred_class_idx = np.argmax(preds[0])
    pred_class = class_names[pred_class_idx]
    confidence = preds[0][pred_class_idx] * 100

    # Display Results
    st.markdown(f"### 🐠 Predicted Category: **{pred_class}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}%**")

    # Show confidence scores for all classes
    st.subheader("Confidence scores for all classes:")
    for label, score in zip(class_names, preds[0]):
        st.write(f"**{label}**: {score*100:.2f}%")
