import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_classifier.keras")

model = load_model()

# ---------------------------
# Class Names
# ---------------------------
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

# ---------------------------
# Title
# ---------------------------
st.title("🖼️ CIFAR-10 Image Classifier")

st.write("""
This application classifies an uploaded image into one of the **10 CIFAR-10 classes**
using a Convolutional Neural Network (CNN) trained with TensorFlow.

### Supported Classes
✈️ Airplane • 🚗 Automobile • 🐦 Bird • 🐱 Cat • 🦌 Deer
🐶 Dog • 🐸 Frog • 🐴 Horse • 🚢 Ship • 🚚 Truck
""")

st.info(
    "Note: The model was trained on 32×32 CIFAR-10 images. "
    "Predictions on high-resolution internet photos may not always be accurate."
)

# ---------------------------
# Upload Image
# ---------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # ---------------------------
    # Preprocess
    # ---------------------------
    resized = image.resize((32, 32))

    img = np.array(resized).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # ---------------------------
    # Prediction
    # ---------------------------
    prediction = model.predict(img, verbose=0)[0]

    top3 = np.argsort(prediction)[-3:][::-1]

    st.success(f"### Final Prediction: {class_names[top3[0]]}")

    st.write("## Top 3 Predictions")

    for i in top3:
        st.write(
            f"**{class_names[i]}** : {prediction[i]*100:.2f}%"
        )

    # ---------------------------
    # Probability Chart
    # ---------------------------
    st.write("## Prediction Probability")

    df = pd.DataFrame({
        "Class": class_names,
        "Probability": prediction
    })

    st.bar_chart(
        df.set_index("Class")
    )

    st.write("---")

    st.caption(
        "Developed using TensorFlow, Keras and Streamlit."
    )
