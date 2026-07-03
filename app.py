import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_classifier.keras")

model = load_model()

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

st.title("🖼️ CIFAR-10 Image Classifier")

st.write(
    "Upload an image and the trained CNN model will predict "
    "which CIFAR-10 class it belongs to."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((32, 32))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    pred = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{class_names[pred]}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
