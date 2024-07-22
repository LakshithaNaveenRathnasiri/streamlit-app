import streamlit as st
import subprocess
import sys

# Function to install packages
def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        st.error(f"Error installing {package}: {e}")
        raise

# Install required packages
for package in ["tensorflow", "matplotlib"]:
    try:
        __import__(package)
    except ImportError:
        install(package)
        __import__(package)

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('image_classification_model.h5')

# Class labels for your model
class_labels = ["Class 1", "Class 2"]  # Replace with your actual class labels

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert('RGB')  # Convert image to RGB if not already
    image = image.resize((224, 224))  # Resize to 224x224 as expected by MobileNetV2
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)

# Function to decode predictions
def decode_predictions(predictions):
    predicted_class_indices = np.argmax(predictions, axis=1)
    return [(class_labels[i], predictions[0][i]) for i in predicted_class_indices]

# Function to display a bar chart of predictions
def plot_predictions(predictions):
    labels = [label for label, _ in predictions]
    scores = [score for _, score in predictions]
    fig, ax = plt.subplots()
    ax.barh(labels, scores)
    ax.set_xlabel('Probability')
    ax.set_title('Top Predictions')
    st.pyplot(fig)

# Title of the Streamlit app
st.title("Image Recognition System")

# Sidebar for additional options
st.sidebar.header("Options")
option = st.sidebar.selectbox("Choose an option:", ["Image Classification", "About"])

# Containers to organize different sections
with st.container():
    st.header("Image Recognition System")
    st.write("This application uses a pre-trained MobileNetV2 model to classify images.")

# File uploader for image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    with st.spinner('Processing...'):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions)
        
    st.success('Done!')

    # Display predictions
    for i, (label, score) in enumerate(decoded_predictions):
        st.write(f"{i + 1}: {label} ({score:.2f})")

    # Plot predictions
    plot_predictions(decoded_predictions)
