import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("cifar10_classifier.keras")

# -----------------------------
# Class Names
# -----------------------------
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

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<h1 style='text-align:center;'>🖼️ CIFAR-10 Image Classifier</h1>

<p style='text-align:center;font-size:18px;color:gray;'>
Classify images into one of the 10 CIFAR-10 categories using a
Convolutional Neural Network (CNN) trained with TensorFlow.
</p>
""", unsafe_allow_html=True)

# -----------------------------
# Supported Classes
# -----------------------------
with st.expander("📋 Supported Classes"):
    st.markdown("""
- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐶 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck
""")

st.info(
    "ℹ️ This model was trained on **32×32 CIFAR-10 images**. "
    "Predictions on high-resolution real-world photos may not always be accurate."
)

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    img = image.resize((32, 32))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index] * 100)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.success(f"### ✅ {predicted_class}")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(min(int(confidence), 100))

    st.markdown("---")

    st.subheader("📊 Top 5 Predictions")

    top5 = np.argsort(prediction[0])[::-1][:5]

    for i in top5:
        st.write(
            f"**{class_names[i]}** — {prediction[0][i]*100:.2f}%"
        )
        st.progress(min(int(prediction[0][i]*100), 100))

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "Developed using TensorFlow • Streamlit • CIFAR-10 Dataset"
)
