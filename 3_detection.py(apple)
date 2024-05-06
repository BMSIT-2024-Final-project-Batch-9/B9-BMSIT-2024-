import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained VGG16 model
model = load_model('mango.keras')

def predict_fruit_disease(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.  # Rescale to [0,1]

    # Predict the class probabilities
    predictions = model.predict(img_array)

    # Convert probabilities to class labels
    class_labels = ['Alternaria', 'Anthracnose', 'Black Mould Rot', 'Healthy', 'Stem end Rot']
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]

    return predicted_class

# Example usage
image_path = str(input("enter image path : "))
predicted_label = predict_fruit_disease(image_path)
print("Predicted class label:", predicted_label)
