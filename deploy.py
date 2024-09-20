import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model = load_model('AeroplaneDamageDetection.h5', compile=False)

model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# model = load_model('AeroplaneDamageDetection.h5')

def predict_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0)  

    prediction = model.predict(image)
    return prediction

st.image('bgimage.png')
st.subheader('By Team8848(TataSafeguard)')
st.title('CNN Based Machine Learning Prediction Model')
st.subheader('Airplane Dents And Damage Detection')
st.write("Here, we have used the dataset and analyze the basis prediction by using ML algorithm, where 4169 images is used, to train our model. This model can analyze dents, cracks, or both in different parts of the airplane.")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

st.page_link('https://github.com/bishalrauniyar/TataSafeguard-MLprediction-damage-detection-model', label='Github', icon='🔗')
if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):

        prediction = predict_image(image)
        class_names = ['Both', 'Crack', 'Dent']
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

    
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Accuracy: {confidence*100}%")

