import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Set page config
st.set_page_config(
    page_title="Pest Classification App",
    page_icon="🐛",
    layout="centered"
)

# Load model and classes
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('mobilenetv2_model.keras')
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        # Reverse mapping from index to class name
        classes = [None] * len(class_indices)
        for k, v in class_indices.items():
            classes[v] = k
        return model, classes
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Image preprocessing
@st.cache_data
def preprocess_image(image):
    img = image.resize((160, 160))
    img = np.array(img) / 255.0
    if img.shape[-1] == 4:
        img = img[..., :3]  # Remove alpha channel if present
    img = np.expand_dims(img, axis=0)
    return img

# Main app
def main():
    st.title("🐛 Pest Classification App")
    st.write("Upload an image of a pest to classify it!")

    # Load model
    model, classes = load_model()
    if model is None or classes is None:
        st.error("Please train the model first by running train_tensorflow.py")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Make prediction
            if st.button("Classify"):
                with st.spinner("Classifying..."):
                    # Preprocess image
                    input_tensor = preprocess_image(image)
                    # Get prediction
                    predictions = model.predict(input_tensor, verbose=0)[0]
                    predicted_class = np.argmax(predictions)
                    confidence = predictions[predicted_class]

                    # Display results
                    st.success(f"Prediction: {classes[predicted_class]}")
                    
                    # Create a progress bar for confidence
                    st.progress(float(confidence))
                    st.write(f"Confidence: {confidence:.2%}")

                    # Add some information about the prediction
                    if confidence > 0.8:
                        st.info("High confidence prediction!")
                    elif confidence < 0.5:
                        st.warning("Low confidence prediction. You might want to try another image.")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 
