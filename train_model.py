"""
train_model.py – Train the CNN model for handwritten digit and operator recognition.

This script reads labeled 28x28 grayscale images from the 'dataset/' folder,
trains a custom CNN, and saves the weights to 'calculator_brain.h5'.

Dataset folder structure:
    dataset/
    ├── 0/
    ├── 1/
    ├── ...
    ├── 9/
    ├── divide/
    ├── minus/
    ├── mult/
    └── plus/
"""

import tensorflow as tf
from tensorflow.keras import layers, models

DATASET_DIR = "dataset"
MODEL_OUTPUT = "calculator_brain.h5"
IMAGE_SIZE = (28, 28)
BATCH_SIZE = 32
EPOCHS = 10

# ============================================================
#  1. LOAD DATASET
# ============================================================
print("Scanning image folders...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"\nDetected classes (alphabetical): {class_names}\n")

# Normalize pixel values from [0, 255] to [0, 1]
rescale = layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (rescale(x), y))
val_ds = val_ds.map(lambda x, y: (rescale(x), y))

# ============================================================
#  2. BUILD THE CNN ARCHITECTURE
# ============================================================
print("Building CNN architecture (14-class classifier)...")

model = models.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    # Fully connected layer
    layers.Dense(128, activation='relu'),

    # Output layer: 14 classes (digits 0-9 + four operators)
    layers.Dense(14, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
#  3. TRAIN
# ============================================================
print(f"Training for {EPOCHS} epochs...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ============================================================
#  4. SAVE
# ============================================================
model.save(MODEL_OUTPUT)
print(f"\nModel saved to '{MODEL_OUTPUT}' successfully.")
