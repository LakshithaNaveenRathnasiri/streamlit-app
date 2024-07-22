import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

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

# Function to generate textual description of the image
def generate_description(predictions):
    descriptions = []
    for _, label, score in predictions:
        description = f"Label: {label}, Score: {score:.2f}"
        descriptions.append(description)
    return descriptions

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
st.title("Image Classification App")

# Sidebar for additional options
st.sidebar.header("Options")
option = st.sidebar.selectbox("Choose an option:", ["Image Classification", "About"])

if option == "Image Classification":
    # Container for image classification
    with st.container():
        st.header("Upload an Image for Classification")
        st.write("Upload an image to classify it using MobileNetV2.")

        # File uploader for image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("Classifying...")

            with st.spinner('Processing...'):
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)
                decoded_predictions = decode_predictions(predictions)
                descriptions = generate_description(decoded_predictions)
                
            st.success('Done!')

            # Display predictions
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                st.write(f"{i + 1}: {label} ({score:.2f})")

            # Display textual descriptions
            st.subheader("Image Description")
            for description in descriptions:
                st.write(description)

            # Plot predictions
            plot_predictions(decoded_predictions)

elif option == "About":
    # Container for about section
    with st.container():
        st.header("About")
        st.write("This application uses a pre-trained MobileNetV2 model to classify images.")
        st.write("Upload an image, and the model will predict the most likely classes along with their probabilities.")
        st.write("Created with Streamlit and TensorFlow.")
