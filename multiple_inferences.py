import argparse
import os
import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def classify_image(image_path, model, trained_class_names):
    """
    Classify a single image using a trained model.

    Args:
        image_path (str): Path to the image to classify.
        model (Model): Loaded model instance.
        trained_class_names (list): List of class names corresponding to model outputs.

    Returns:
        str: Predicted class name.
    """
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

    return predicted_class_name

def classify_from_csv(csv_path, model_path, batch_size=32):
    """
    Classify images listed in a CSV file using batch processing.

    Args:
        csv_path (str): Path to the CSV file containing image paths.
        model_path (str): Path to the trained model file.
        batch_size (int): Number of images to process in each batch.

    Returns:
        pd.DataFrame: DataFrame with original columns and predicted class names.
    """
    # Load the model
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Define class names
    trained_class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

    # Load the CSV file
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if "image_path" not in data.columns:
        raise ValueError("The CSV file must contain an 'image_path' column.")

    # Add predicted class to each row
    predictions = []

    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        images = []
        batch_paths = batch["image_path"].tolist()

        for image_path in batch_paths:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Image file not found: {image_path}")
                predictions.extend(["Error: Image not found"] * len(batch_paths))
                continue
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            images.append(image)

        if images:
            images = np.array(images, dtype=np.float32)
            raw_predictions = model.predict(images)
            predicted_indices = np.argmax(raw_predictions, axis=1)
            batch_predictions = [trained_class_names[idx] for idx in predicted_indices]
            predictions.extend(batch_predictions)

    data["predicted_class"] = predictions
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify images from a CSV file using a trained model.")
    parser.add_argument("-c", "--csv", type=str, required=True,
                        help="Path to the CSV file containing image paths.")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Path to the trained model file.")
    parser.add_argument("-o", "--output", type=str, required=False, default="predictions.csv",
                        help="Path to save the output CSV with predictions (default: 'predictions.csv').")

    args = parser.parse_args()

    # Classify images from the CSV
    results_df = classify_from_csv(args.csv, args.model)

    # Save the results to a CSV file
    results_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}.")
