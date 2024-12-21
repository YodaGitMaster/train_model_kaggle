import numpy as np
import pandas as pd
import cv2
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import tensorflow as tf


# ----------------------------
# Set Random Seeds for Reproducibility
# ----------------------------
def set_random_seed(seed=123):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ----------------------------
# Load Class Names
# ----------------------------
def get_class_names(train_path):
    """
    Get sorted class names from the training directory.
    """
    return sorted(os.listdir(train_path))


# ----------------------------
# Load and Preprocess Train Data
# ----------------------------
def load_train_data(train_path, class_names):
    """
    Load and preprocess training images.
    """
    X = []
    y = []
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(train_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            X.append(image / 255.0)  # Normalize
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ----------------------------
# Analyze Class Distribution
# ----------------------------
def log_class_distribution(y, class_names):
    """
    Analyze and log class distribution.
    """
    class_counts = pd.Series(y).value_counts().sort_index()
    print("\nClass Distribution:")
    for i, count in enumerate(class_counts):
        print(f"{class_names[i]}: {count}")

    # Plot class distribution
    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar', title='Class Distribution', color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Split Data
# ----------------------------
def split_data(X, y, test_size=0.2, seed=123):
    """
    Split data into training and validation sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# ----------------------------
# Data Augmentation
# ----------------------------
def create_data_augmentation():
    """
    Create a data augmentation pipeline.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])


# ----------------------------
# Define Model
# ----------------------------
def build_model(class_names, data_augmentation):
    """
    Build the EfficientNetV2S model with custom output layer.
    """
    base_model = keras.applications.EfficientNetV2S(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False  # Freeze the base model

    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, base_model


# ----------------------------
# Train Model
# ----------------------------
def train_model(model, X_train, y_train, X_val, y_val, class_weights=None, epochs=25):
    """
    Train the model with early stopping.
    """
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        epochs=epochs,
        callbacks=[early_stop]
    )
    return history


# ----------------------------
# Fine-Tune Model
# ----------------------------
def fine_tune_model(model, base_model, X_train, y_train, X_val, y_val, epochs=10):
    """
    Fine-tune the model by unfreezing the base model.
    """
    base_model.trainable = True  # Unfreeze the base model
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[early_stop])


# ----------------------------
# Evaluate Model
# ----------------------------
def evaluate_model(model, X_val, y_val, class_names):
    """
    Evaluate the model and print a classification report.
    """
    val_predictions = np.argmax(model.predict(X_val), axis=1)
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_predictions, target_names=class_names))


# ----------------------------
# Save Model
# ----------------------------
def save_model(model, output_path="best_model.h5"):
    """
    Save the trained model to disk.
    """
    model.save(output_path)
    print(f"Model saved to {output_path}")


# ----------------------------
# Main Execution
# ----------------------------
def main():
    TRAIN_PATH = "train_images"  # Update this to your training data path
    TEST_PATH = "test_images"    # Update this to your test data path

    # Set random seed
    set_random_seed()

    # Load class names
    class_names = get_class_names(TRAIN_PATH)
    print(f"Classes: {class_names}")

    # Load and analyze training data
    X, y = load_train_data(TRAIN_PATH, class_names)
    log_class_distribution(y, class_names)

    # Split data
    X_train, X_val, y_train, y_val = split_data(X, y)
    del X, y  # Free memory

    # Compute class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    print("\nClass Weights:", class_weights)

    # Create data augmentation pipeline
    data_augmentation = create_data_augmentation()

    # Build model
    model, base_model = build_model(class_names, data_augmentation)

    # Train the model
    print("\nStarting initial training...")
    train_model(model, X_train, y_train, X_val, y_val, class_weights=class_weights)

    # Fine-tune the model
    print("\nFine-tuning the model...")
    fine_tune_model(model, base_model, X_train, y_train, X_val, y_val)

    # Evaluate the model
    evaluate_model(model, X_val, y_val, class_names)

    # Save the model
    save_model(model)


if __name__ == "__main__":
    main()
