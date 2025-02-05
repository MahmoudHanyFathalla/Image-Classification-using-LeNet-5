import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
loaded_model = tf.keras.models.load_model("C:\\Users\\hp\\Desktop\\KoraState\\task\\lenet_model.h5")

# Load a single test image
test_image_path = "C:\\Users\\hp\\Desktop\\KoraState\\task\\8.jpeg"

try:
    with Image.open(test_image_path) as img:
        # Convert image to NumPy array
        img_array = np.array(img)

        # Ensure the array has the expected dimensions
        if img_array.ndim == 2:
            # Normalize pixel values to be between 0 and 1
            img_array = img_array.astype('float32') / 255.0

            # Reshape the array to match the input shape of the model
            img_array = np.reshape(img_array, (1, 28, 28, 1))

            # Make predictions
            predictions = loaded_model.predict(img_array)

            # Get the predicted label (index with the highest probability)
            predicted_label = np.argmax(predictions)

            print(f"Predicted Label: {predicted_label}")
        else:
            print("Image has unexpected dimensions and cannot be processed.")
except Exception as e:
    print(f"Error loading image '{test_image_path}': {e}")
