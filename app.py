import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image

# Load model and butterfly info
model = tf.keras.models.load_model("butterfly_model.h5")

with open("butterfly_info.json", "r") as f:
    butterfly_info = json.load(f)

# Set Streamlit page title
st.set_page_config(page_title="🦋 Enchanted Wings - Butterfly Classifier")

st.title("🦋 Enchanted Wings: Butterfly Species Classifier")
st.write("Upload an image of a butterfly and discover its species!")

# Upload image
uploaded_file = st.file_uploader("Choose a butterfly image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict species
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    class_labels = list(butterfly_info.keys())
    predicted_label = class_labels[predicted_index]

    # Show prediction and description
    st.subheader("Prediction:")
    st.markdown(f"**🦋 {butterfly_info[predicted_label]['name']}**")
    st.write(butterfly_info[predicted_label]["description"])
