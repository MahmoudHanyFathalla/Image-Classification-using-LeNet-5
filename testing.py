import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

current_positions = {}  # Keep track of the current position for each folder

def distribute_batch_sizes(total_batch_size, num_folders):
   
    if num_folders == 0:
        raise ValueError("Number of folders cannot be zero.")

    batch_size_per_folder = total_batch_size // num_folders
    remaining_batch_size = total_batch_size % num_folders

    batch_sizes = {f"{i}_imgs": batch_size_per_folder for i in range(num_folders)}

    # Distribute the remaining batch size equally among the folders
    for i in range(remaining_batch_size):
        batch_sizes[f"{i}_imgs"] += 1

    return batch_sizes

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    # Convert image to NumPy array
                    img_array = np.array(img)
                    
                    # Check if the array has the expected dimensions
                    if img_array.ndim == 2:
                        images.append(img_array)
                    else:
                        print(f"Image '{img_path}' has unexpected dimensions and cannot be added.")
            except Exception as e:
                print(f"Error loading image '{img_path}': {e}")
    return images


def assign_labels(img_folder):
    # Assign labels based on the index of the folder in img_folders
    return int(img_folder.split('_')[0])

def batchloader(folder_path, batch_sizes):
    global current_positions

    img_folders = [f for f in os.listdir(folder_path) if f.endswith("_imgs")]

    img_batch = []
    label_batch = []

    for img_folder in img_folders:
        img_folder_path = os.path.join(folder_path, img_folder)

        images = load_images_from_folder(img_folder_path)
        label = assign_labels(img_folder)

        if img_folder not in current_positions:
            current_positions[img_folder] = 0

        batch_size = batch_sizes.get(img_folder, 0)
        start_position = current_positions[img_folder]
        end_position = start_position + batch_size

        img_batch.extend(images[start_position:end_position])
        label_batch.extend([label] * len(images[start_position:end_position]))

        # Move to the next set of images
        current_positions[img_folder] = end_position

    return img_batch, label_batch




# Load the saved model
loaded_model = tf.keras.models.load_model("C:\\Users\\hp\\Desktop\\KoraState\\task\\lenet_model.h5")

# Use the batch loader function for testing
test_folder_path = "C:\\Users\\hp\\Desktop\\KoraState\\task\\test"
test_batch_sizes = distribute_batch_sizes(20, 10)  # Adjust batch size as needed

img_batch, label_batch = batchloader(test_folder_path, test_batch_sizes)

# Convert img_batch to NumPy array and normalize pixel values
x_test = np.array(img_batch).astype('float32') / 255.0

# Make predictions
predictions = loaded_model.predict(x_test)

# Evaluate the model on the test data
accuracy = loaded_model.evaluate(x_test, np.array(label_batch), verbose=0)[1]
print(f"Test Accuracy: {accuracy}")
