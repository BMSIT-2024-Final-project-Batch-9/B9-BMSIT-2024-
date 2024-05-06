import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image

# Load the trained model
model = load_model('mango.keras')

# Define image dimensions
img_width, img_height = 224, 224

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def classify_image(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(prediction)
    classes = ['Alternaria', 'Anthracnose', 'Black Mould Rot', 'Healthy', 'Stem end Rot']  # Update with your class labels
    predicted_class = classes[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    return predicted_class, confidence

# Example usage
image_path = str(input('enter image path : '))
predicted_class, confidence = classify_image(image_path)
print("Predicted Class:", predicted_class)
print("Confidence:", confidence)