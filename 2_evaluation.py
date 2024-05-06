import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define data directory for test dataset
test_dir = r'C:\Users\naman\OneDrive\Desktop\fruit disease\pythonProject\2_mango\mango_image_dataset\test'

# Define image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Data normalization
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow test images in batches using test_datagen generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Keep the order of test images
)

# Load the saved best model
model = tf.keras.models.load_model('2_mango.keras')

# Evaluate the model on the test dataset
eval_result = model.evaluate(test_generator, verbose=1)
print("Test Loss:", eval_result[0])
print("Test Accuracy:", eval_result[1])

# Get the true labels and predicted probabilities
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)

# Get the predicted labels
y_pred = np.argmax(y_pred_prob, axis=1)

# Generate classification report
print("MANGO DENSENET CLASSIFICATION REPORT : \n",classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Mango DenseNet Classification Report')
plt.savefig('mango_confusion_matrix.png',bbox_inches='tight')
plt.show()