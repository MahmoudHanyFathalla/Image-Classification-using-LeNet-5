# Image Classification using LeNet-5

## Overview

This project focuses on image classification using the LeNet-5 convolutional neural network (CNN) architecture. The goal is to classify images into predefined categories by training a deep learning model on a dataset of labeled images. The project includes data loading, model training, and evaluation, as well as inference on single images.

## Features

- **LeNet-5 Model**: The project uses the LeNet-5 architecture, a classic CNN model, for image classification.
- **Data Loading and Preprocessing**: Functions to load images from folders, assign labels, and prepare batches for training and testing.
- **Training**: The model is trained on a dataset of images, with support for distributed batch sizes across multiple folders.
- **Inference**: The trained model can be used to classify single images or batches of images.
- **Evaluation**: The model's performance is evaluated on a test dataset, and accuracy is reported.

## Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.x**
- **TensorFlow**
- **NumPy**
- **Pillow (PIL)**
- **Matplotlib** (optional, for visualization)

You can install the required Python libraries using pip:

```bash
pip install tensorflow numpy pillow matplotlib
```

## Project Structure

The project consists of the following files:

- **dataloader.py**: Contains functions for loading images, assigning labels, and preparing batches for training. It also includes the LeNet-5 model definition and training loop.
- **oneimgtest.py**: A script to load a single image and perform inference using the trained model.
- **testing.py**: A script to evaluate the model's performance on a test dataset.

## Usage

### Data Loading and Preprocessing

The `dataloader.py` script includes functions to load images from folders, assign labels, and prepare batches for training. The `load_images_from_folder()` function loads images from a specified folder, while the `batchloader()` function prepares batches of images and labels for training.

```python
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    img_array = np.array(img)
                    if img_array.ndim == 2:
                        images.append(img_array)
                    else:
                        print(f"Image '{img_path}' has unexpected dimensions and cannot be added.")
            except Exception as e:
                print(f"Error loading image '{img_path}': {e}")
    return images

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

        current_positions[img_folder] = end_position

    return img_batch, label_batch
```

### Model Definition

The LeNet-5 model is defined in the `create_lenet_model()` function. The model consists of two convolutional layers, two max-pooling layers, and three fully connected layers.

```python
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
```

### Training

The model is trained using the `batchloader()` function to prepare batches of images and labels. The training loop is implemented in the `dataloader.py` script.

```python
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
```

### Inference on Single Image

The `oneimgtest.py` script allows you to load a single image and perform inference using the trained model.

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
loaded_model = tf.keras.models.load_model("C:\\Users\\hp\\Desktop\\KoraState\\task\\lenet_model.h5")

# Load a single test image
test_image_path = "C:\\Users\\hp\\Desktop\\KoraState\\task\\8.jpeg"

try:
    with Image.open(test_image_path) as img:
        img_array = np.array(img)
        if img_array.ndim == 2:
            img_array = img_array.astype('float32') / 255.0
            img_array = np.reshape(img_array, (1, 28, 28, 1))
            predictions = loaded_model.predict(img_array)
            predicted_label = np.argmax(predictions)
            print(f"Predicted Label: {predicted_label}")
        else:
            print("Image has unexpected dimensions and cannot be processed.")
except Exception as e:
    print(f"Error loading image '{test_image_path}': {e}")
```

### Model Evaluation

The `testing.py` script evaluates the model's performance on a test dataset. It calculates the accuracy of the model on the test data.

```python
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
```

## Conclusion

This project provides a comprehensive solution for image classification using the LeNet-5 architecture. It includes data loading, model training, inference, and evaluation, making it a complete solution for image classification tasks. The project is designed to be flexible and can be adapted to different datasets and classification tasks.
