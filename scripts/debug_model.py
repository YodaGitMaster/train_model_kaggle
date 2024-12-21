import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# Paths and Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_testdata_file(testdata_path):
    """
    Load test data and verify paths.
    """
    if not os.path.exists(testdata_path):
        raise FileNotFoundError(f"Test data file not found: {testdata_path}")

    data = pd.read_csv(testdata_path)
    required_columns = {'image_path', 'class_name', 'label'}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in test data file: {missing_columns}")

    # Verify image paths
    data['exists'] = data['image_path'].apply(os.path.exists)
    invalid_paths = data.loc[~data['exists'], 'image_path']
    if not invalid_paths.empty:
        print(f"Warning: Some image paths do not exist:\n{invalid_paths}")
    data = data[data['exists']].copy()
    data.drop(columns=['exists'], inplace=True)
    return data


def verify_model_mapping(model, trained_class_names):
    """
    Verify the mapping between model output indices and class names.
    """
    print("Verifying model class mapping...")
    num_classes = len(trained_class_names)
    simulated_output = np.eye(num_classes)  # Simulate one-hot outputs

    for idx, row in enumerate(simulated_output):
        predicted_index = np.argmax(row)
        predicted_label = trained_class_names[predicted_index]
        print(f"Simulated Output {row} -> Predicted Index: {predicted_index}, Predicted Label: {predicted_label}")


def test_model_on_sample_images(model, trained_class_names, sample_images):
    """
    Test model predictions on sample images for debugging.
    """
    print("\nTesting model on sample images...")
    for img_path in sample_images:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image not found: {img_path}")
            continue

        # Preprocess image
        image = cv2.resize(image, (224, 224))  # Adjust to model input size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(image)
        predicted_index = np.argmax(prediction)
        predicted_label = trained_class_names[predicted_index]

        print(f"Image: {img_path}")
        print(f"Raw Prediction: {prediction}")
        print(f"Predicted Index: {predicted_index}, Predicted Label: {predicted_label}\n")


def analyze_training_data(training_data_path):
    """
    Analyze training data for class distribution.
    """
    print("\nAnalyzing training data...")
    data = pd.read_csv(training_data_path)
    if 'class_name' not in data.columns:
        raise ValueError("Training data must have a 'class_name' column.")

    class_distribution = data['class_name'].value_counts()
    print("Training data class distribution:")
    print(class_distribution)

    # Plot class distribution
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_distribution.index, y=class_distribution.values)
    plt.title("Training Data Class Distribution")
    plt.xlabel("Class Name")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    distribution_plot = os.path.join(RESULTS_DIR, "training_class_distribution.png")
    plt.savefig(distribution_plot)
    print(f"Class distribution plot saved to {distribution_plot}")


def predict_and_save(test_data, model, trained_class_names, output_folder):
    """
    Perform predictions and save results.
    """
    print("Starting inference...")
    images = []
    for img_path in test_data['image_path']:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image file not found: {img_path}")
            continue
        image = cv2.resize(image, (224, 224))  # Resize to model input size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0  # Normalize
        images.append(image)

    images = np.array(images, dtype=np.float32)

    # Predict
    raw_predictions = model.predict(images)
    predictions = np.argmax(raw_predictions, axis=1)
    predicted_labels = [trained_class_names[idx] for idx in predictions]

    # Save predictions
    test_data['prediction'] = predictions
    test_data['predicted_class_name'] = predicted_labels
    predictions_file = os.path.join(output_folder, "predictions.csv")
    test_data.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Debug model predictions and validate mappings."
    )
    parser.add_argument(
        '-m', '--model_path', type=str, required=True,
        help="Path to the trained model file."
    )
    parser.add_argument(
        '-t', '--testdata_path', type=str, required=True,
        help="Path to the test data CSV file."
    )
    parser.add_argument(
        '-r', '--training_data_path', type=str, required=False,
        help="Path to the training data CSV file for class distribution analysis."
    )
    parser.add_argument(
        '-o', '--output_folder', type=str, required=False, default="results",
        help="Folder to save debugging outputs."
    )
    args = parser.parse_args()

    # Load model
    try:
        model = load_model(args.model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Model file not found at {args.model_path}")
        exit(1)

    # Define class names
    trained_class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

    # Verify model mapping
    verify_model_mapping(model, trained_class_names)

    # Load test data
    test_data = load_testdata_file(args.testdata_path)

    # Debugging: Test predictions on a few sample images
    sample_images = [
        "train_images/daisy/5722473541_ffac1ae67e_n.jpg",
        "train_images/dandelion/20165867412_fc45d31698_m.jpg",
        "train_images/rose/6209630964_e8de48fe04_m.jpg",
        "train_images/tulip/4516198427_0e5099cd8e.jpg",
        "train_images/sunflower/4846786944_2832c5c8b8.jpg"
    ]
    test_model_on_sample_images(model, trained_class_names, sample_images)

    # Analyze training data if provided
    if args.training_data_path:
        analyze_training_data(args.training_data_path)

    # Predict and save results
    output_folder = os.path.join(RESULTS_DIR, "debugging")
    os.makedirs(output_folder, exist_ok=True)
    predict_and_save(test_data, model, trained_class_names, output_folder)


if __name__ == "__main__":
    main()
