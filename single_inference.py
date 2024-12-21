import argparse
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def classify_image(image_path, model_path):
    """
    Classify a single image using a trained model.

    Args:
        image_path (str): Path to the image to classify.
        model_path (str): Path to the trained model file.

    Returns:
        str: Predicted class name.
    """
    # Load the model
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Define class names
    trained_class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform inference
    raw_prediction = model.predict(image)
    predicted_index = np.argmax(raw_prediction, axis=1)[0]
    predicted_class_name = trained_class_names[predicted_index]

    print(f"Predicted class: {predicted_class_name}")
    return predicted_class_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify a single image using a trained model.")
    parser.add_argument("-i", "--image", type=str, required=True,
                        help="Path to the image to classify.")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Path to the trained model file.")

    args = parser.parse_args()

    # Perform classification
    classify_image(args.image, args.model)
