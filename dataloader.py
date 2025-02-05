import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

current_positions = {}  # Keep track of the current position for each folder

def create_lenet_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


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


'''
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return images

'''

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




# Set your input shape and number of classes
input_shape = (28, 28, 1)  # Replace with actual values
num_classes = 10  # Replace with the actual number of classes

# Create and compile your LeNet model
model = create_lenet_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set your folder path, total batch size, and number of folders
folder_path = "C:\\Users\\hp\\Desktop\\KoraState\\task\\data"
total_batch_size = 6000
num_folders = 10

# Distribute batch sizes
batch_sizes = distribute_batch_sizes(total_batch_size, num_folders)

# Train your model using the batch loader function
epochs = 10  # Replace with the desired number of epochs

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    img_batch, label_batch = batchloader(folder_path, batch_sizes)

    # Convert img_batch and label_batch to NumPy arrays
    x_train = np.array(img_batch)
    y_train = np.array(label_batch)

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0

    # Train the model
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1)


# Save the trained model with a complete file path
model.save("C:\\Users\\hp\\Desktop\\KoraState\\task\\lenet_model.h5")
print("Model saved successfully.")



