# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
import cv2
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image

def main():

    class_names = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']

    st.title('Predicting an Image based on six different categories')

    st.markdown("This is a model based on Tensorflow and Keras for classification of images.")
    st.markdown("Depending on the image you upload, the model will try to categorize it in one of six options available.")

    img_building = Image.open('baseImages/building.jpg')
    img_forest = Image.open('baseImages/forest.jpg')
    img_glacier = Image.open('baseImages/glacier.jpg')
    img_mountain = Image.open('baseImages/mountain.jpg')
    img_sea = Image.open('baseImages/sea.jpg')
    img_street = Image.open('baseImages/street.jpg')

    col1, col2, col3 = st.beta_columns(3)
    col4, col5, col6 = st.beta_columns(3)

    col1.header("Building")
    col1.image(img_building)

    col2.header("Forest")
    col2.image(img_forest)

    col3.header("Glacier")
    col3.image(img_glacier)

    col4.header("Mountain")
    col4.image(img_mountain)

    col5.header("Sea")
    col5.image(img_sea)

    col6.header("Street")
    col6.image(img_street)

    st.markdown("Upload an image similar to the ones mentioned above, the image will be resized and showed to you along with to possible classifications")

    upload_image = st.file_uploader("Choose a File",type=["png","jpg","jpeg"])

    model_xception = load_pretrained_model()

    batch_size = 32
    if upload_image is not None:
        bytes_data = upload_image.getvalue()
        data = np.frombuffer(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(data,1)
        img = cv2.resize(img, (150, 150))  # Resize the image
        st.image(img)
        np_img = np.array(img)
        np_img = np_img[np.newaxis, ...]
        img_dataset = tf.data.Dataset.from_tensor_slices(np_img)
        img_dataset = img_dataset.map(preprocess_prediction).batch(batch_size)
        prediction = model_xception.predict(img_dataset)
        prediction = np.ravel(prediction)
        idx_best_pred = prediction.argsort()[-2:][::-1]
        st.markdown("The top 2 classifications are:")
        for index in idx_best_pred[-2:]:
            st.markdown(class_names[index])


def preprocess_prediction(image):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image

@st.cache
def load_pretrained_model():
    model = keras.models.load_model('model/IMG_intel.h5')
    return model

if __name__ == '__main__':
    main()
