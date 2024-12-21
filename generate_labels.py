import argparse
import os
import pandas as pd

def generate_labels_csv(train_path, output_folder=None):
    """
    Generates a CSV file mapping images to their class labels.

    Args:
        train_path (str): Path to the training images folder.
        output_folder (str): Optional. Folder to save the output CSV. Defaults to the current directory.
    """
    class_names = sorted(os.listdir(train_path))
    data = []

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(train_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            data.append((image_path, class_name, label))

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["image_path", "class_name", "label"])

    # Default output folder to current directory if not specified
    if output_folder is None:
        output_folder = os.getcwd()

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the labels.csv
    labels_csv_path = os.path.join(output_folder, "labels.csv")
    df.to_csv(labels_csv_path, index=False)
    print(f"Labels CSV saved to {labels_csv_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate a labels CSV for training data.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to the training images folder.")
    parser.add_argument("--output_folder", type=str,
                        help="Optional. Folder to save the output CSV.")

    args = parser.parse_args()

    # Generate the labels CSV with the provided arguments
    generate_labels_csv(args.train_path, args.output_folder)
