import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ----------------------------
# Enable GPU Usage
# ----------------------------


def enable_gpu():
    """
    Verify and enable GPU for TensorFlow. Exit if no GPU is detected.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(
                gpu, True)  # Prevent memory lock issues
    else:
        print("No GPU detected. Exiting...")
        exit(1)


# ----------------------------
# Set Random Seeds for Reproducibility
# ----------------------------


def set_random_seed(seed=123):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ----------------------------
# Define Model
# ----------------------------


def build_model(class_names, data_augmentation):
    """
    Build the EfficientNetV2S model with custom output layer.
    """
    base_model = tf.keras.applications.EfficientNetV2S(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False  # Freeze the base model

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, base_model

# ----------------------------
# Main Execution
# ----------------------------


def main():
    TRAIN_PATH = "train_images"  # Update this to your training data path
    TEST_PATH = "test_images"    # Update this to your test data path

    # Enable GPU
    enable_gpu()

    # Set random seed
    set_random_seed()

    # Load class names
    class_names = sorted(os.listdir(TRAIN_PATH))
    print(f"Classes: {class_names}")

    # Load and preprocess training data
    X, y = [], []
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(TRAIN_PATH, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            X.append(image / 255.0)
            y.append(label)
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y)

    # Compute class weights
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    print("\nClass Weights:", class_weights)

    # Create data augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    # Build and train model
    model, base_model = build_model(class_names, data_augmentation)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        epochs=25,
        callbacks=[early_stop]
    )

    # Fine-tune the model
    base_model.trainable = True  # Unfreeze base model
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(
        X_val, y_val), epochs=10, callbacks=[early_stop])

    # Evaluate the model
    val_predictions = np.argmax(model.predict(X_val), axis=1)
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_predictions, target_names=class_names))

    # Save the model
    model.save("best_model.h5")
    print("Model saved to best_model.h5")


if __name__ == "__main__":
    main()
