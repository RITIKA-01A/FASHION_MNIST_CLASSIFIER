import streamlit as st
import tensorflow as tf
import keras
from PIL import Image
import numpy as np
import time

# Load the trained model
model = keras.models.load_model("MNIST_FASHION_CLFF_MODEL.keras")

# Define class labels
class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape((1, 28, 28))
    return img_array

# --- Streamlit App UI ---
st.set_page_config(page_title="Fashion MNIST Classifier", page_icon="🛍️", layout="wide")

# Custom CSS Styling
st.markdown("""
    <style>
        body {background-color: #FAF3E0; font-family: 'Arial', sans-serif;}
        .title {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            margin-top: -50px;
            background: -webkit-linear-gradient(45deg, #FF5733, #FFC300);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 10px;
        }
        .upload-box {
            border: 2px dashed #666;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            background-color: #FFF8E1;
        }
        .btn-classify {
            background-color: #FF4B4B;
            color: white;
            padding: 10px;
            border-radius: 8px;
            font-size: 18px;
            width: 100%;
        }
        .btn-classify:hover {background-color: #E63946;}
        .result-box {
            border: 2px solid #FF5733;
            padding: 20px;
            border-radius: 10px;
            background-color: #FFF0E1;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# App Title (Positioned Higher)
st.markdown('<h1 class="title">🛍️ Fashion Item Classifier</h1>', unsafe_allow_html=True)

# Upload Image Section
st.markdown('<div class="upload-box"><h4>📤 Upload a fashion image (JPG, JPEG, PNG)</h4></div>', unsafe_allow_html=True)
uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])

# Layout: Left (Image) | Right (Prediction)
if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Uploaded Image")
        resized_image = image.resize((150, 150))  # Compact image display
        st.image(resized_image, caption="Your uploaded fashion item", use_container_width =False)

    with col2:
        st.subheader("🔍 Model Prediction")
        if st.button('🚀 Classify', key="classify_btn"):
            with st.spinner('🧠 Analyzing...'):
                time.sleep(1.5)  # Simulate loading delay
                img_array = preprocess_image(uploaded_image)
                result = model.predict(img_array)

                # Confidence Score
                predicted_class = np.argmax(result)
                confidence = np.max(result) * 100
                prediction = class_labels[predicted_class]

                # Display results in a styled box
                st.markdown(f"""
                    <div class='result-box'>
                        🎯 Prediction: <b>{prediction}</b> <br>
                        📊 Confidence: <b>{confidence:.2f}%</b>
                    </div>
                """, unsafe_allow_html=True) 