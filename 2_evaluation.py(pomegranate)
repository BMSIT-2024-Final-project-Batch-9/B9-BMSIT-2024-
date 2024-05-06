import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories
test_dir = r'C:\Users\naman\OneDrive\Desktop\fruit disease\pythonProject\3_pomegranate\pomegranate_disease_dataset\test'

# Parameters
batch_size = 32
image_size = (150, 150)

# Data preprocessing
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important to keep the same order as predictions
)

# Load the saved model
model = load_model("best_model.weights.keras")

# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)

# Predict classes for test data
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Print classification report
class_labels = list(test_generator.class_indices.keys())
print("Pomegranate Disease Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Plot confusion matrix as green heatmap
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Pomegranate Disease Confusion Matrix')
plt.savefig('pomegranate_confusion_matrix.png',bbox_inches='tight')
plt.show()
