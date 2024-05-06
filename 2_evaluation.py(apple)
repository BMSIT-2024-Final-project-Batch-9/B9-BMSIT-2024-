import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained VGG16 model
model = load_model('mango.keras')

# Define the test directory
test_dir = r'C:\Users\naman\OneDrive\Desktop\fruit disease\pythonProject\2_mango\mango_image_dataset\test'

# Set up data generator for test images
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

# Get true labels and make predictions
true_labels = test_generator.classes
predictions = model.predict(test_generator)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Define class labels
class_labels = ['Alternaria', 'Anthracnose', 'Black Mould Rot', 'Healthy','Stem end Rot']


