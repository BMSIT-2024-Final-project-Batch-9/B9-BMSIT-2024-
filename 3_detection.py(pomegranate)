import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model("best_model.weights.keras")

# Define the path to the test image
test_image_path = str(input("enter image path : "))


# Load and preprocess the test image
img = image.load_img(test_image_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.  # Rescale pixel values to [0, 1]

# Make prediction
prediction = model.predict(img_array)

# Decode prediction
class_indices = {0: 'Alternaria', 1: 'Anthracnose', 2: 'Bacterial_Blight', 3: 'Cercospora', 4: 'Healthy'}  # Update with your class labels
predicted_class = class_indices[np.argmax(prediction)]

print("Predicted class:", predicted_class)
