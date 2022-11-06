# -*- coding: utf-8 -*-
"""img_classification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sWCGHRzkK_X5dAGZLGnRzbi83mYNYYQZ
"""

from google.colab import drive
drive.mount('/content/drive')

import keras
from PIL import Image, ImageOps
import numpy as np


def teachable_machine_classification(img):
    # Load the model
    model = keras.models.load_model('/content/drive/MyDrive/IDU-CV Lab Work/COV19D_2nd - Trnasfer Learning/Saved Models/Modified_Xception.h5')

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability