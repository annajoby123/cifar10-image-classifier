import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>

.main{
    padding-top:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

h1{
    text-align:center;
    color:#4F8BF9;
    font-size:50px;
}

h3{
    color:#4F8BF9;
}

.stButton>button{
    width:100%;
    border-radius:10px;
}

.prediction{
    background:#1f2937;
    padding:20px;
    border-radius:12px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_classifier.keras")

model = load_model()

# ----------------------------
# Class Names
# ----------------------------
class_names = [
    "Airplane ✈️",
    "Automobile 🚗",
    "Bird 🐦",
    "Cat 🐱",
    "Deer 🦌",
    "Dog 🐶",
    "Frog 🐸",
    "Horse 🐴",
    "Ship 🚢",
    "Truck 🚚"
]

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🧠 Project Information")

st.sidebar.markdown("""
### CNN Image Classifier

This project classifies images into **10 CIFAR-10 classes** using a Convolutional Neural Network.

### Dataset
- CIFAR-10
- 60,000 Images
- 10 Classes

### Model
- TensorFlow
- CNN
- Accuracy ≈ **74%**

### Developed By
Anna Joby
""")

# ----------------------------
# Header
# ----------------------------
st.title("🧠 CIFAR-10 Image Classifier")

st.markdown("""
### Deep Learning Based Image Classification

Upload an image and let the trained CNN predict its class.
""")

st.info("""
**Supported Classes**

✈️ Airplane • 🚗 Automobile • 🐦 Bird • 🐱 Cat • 🦌 Deer • 🐶 Dog • 🐸 Frog • 🐴 Horse • 🚢 Ship • 🚚 Truck
""")

# ----------------------------
# Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)

# ----------------------------
# Prediction
# ----------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(image,use_container_width=True)

    img = image.resize((32,32))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img,verbose=0)

    pred = np.argmax(prediction)
    confidence = np.max(prediction)*100

    with col2:

        st.subheader("Prediction")

        st.success(f"### {class_names[pred]}")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(float(confidence)/100)

        st.subheader("Top Predictions")

        top3 = np.argsort(prediction[0])[::-1][:3]

        for i in top3:
            st.write(
                f"**{class_names[i]}** : {prediction[0][i]*100:.2f}%"
            )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")

st.caption(
"""
Developed using TensorFlow, Streamlit and the CIFAR-10 Dataset.
"""
)
