import subprocess

# List installed packages
installed_packages = subprocess.check_output(['pip', 'freeze'])
for package in installed_packages.decode().split('\n'):
    if package:
        print(package)

import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
import matplotlib.pyplot as plt# type: ignore



# Load the trained model
model = tf.keras.models.load_model('image_classification_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to 224x224 as expected by MobileNetV2
    image = np.array(image)
    if image.shape[2] == 4:  # Check for alpha channel and remove it if present
        image = image[:, :, :3]
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)

# Function to decode predictions
def decode_predictions(predictions):
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
    return decoded

# Function to display a bar chart of predictions
def plot_predictions(predictions):
    labels = [label for _, label, _ in predictions]
    scores = [score for _, _, score in predictions]
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
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i + 1}: {label} ({score:.2f})")

    # Plot predictions
    plot_predictions(decoded_predictions)
