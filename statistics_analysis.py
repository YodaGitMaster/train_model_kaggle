import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def evaluate_predictions(csv_path):
    """
    Evaluate the correctness of predictions in a CSV file, generate a confusion matrix, and save plots.

    Args:
        csv_path (str): Path to the CSV file containing image_path, class_name, label, and predicted_class.

    Returns:
        None
    """
    # Load the CSV file
    if not csv_path or not csv_path.endswith('.csv'):
        raise ValueError("Please provide a valid path to a CSV file.")

    data = pd.read_csv(csv_path)

    # Check required columns
    required_columns = ["image_path", "class_name", "predicted_class"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"The CSV file must contain a '{col}' column.")

    # Calculate correctness
    data["correct"] = data["class_name"] == data["predicted_class"]
    accuracy = data["correct"].mean() * 100

    # Print evaluation summary
    print(f"Total Images: {len(data)}")
    print(f"Correct Predictions: {data['correct'].sum()}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Create results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Generate confusion matrix
    confusion = pd.crosstab(data["class_name"], data["predicted_class"], rownames=["Actual"], colnames=["Predicted"], normalize="index")

    # Save confusion matrix as a CSV file
    confusion_csv_path = os.path.join(results_dir, "confusion_matrix.csv")
    confusion.to_csv(confusion_csv_path)
    print(f"Confusion matrix saved to {confusion_csv_path}")

    # Plot confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    confusion_plot_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(confusion_plot_path)
    plt.close()
    print(f"Confusion matrix plot saved to {confusion_plot_path}")

    # Plot accuracy pie chart
    plt.figure(figsize=(6, 6))
    data["correct"].value_counts().plot.pie(autopct="%1.1f%%", labels=["Correct", "Incorrect"], colors=["#4CAF50", "#F44336"], startangle=90)
    plt.title("Prediction Accuracy")
    pie_chart_path = os.path.join(results_dir, "accuracy_pie_chart.png")
    plt.savefig(pie_chart_path)
    plt.close()
    print(f"Accuracy pie chart saved to {pie_chart_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the correctness of predictions in a CSV file and generate reports.")
    parser.add_argument("-c", "--csv", type=str, required=True,
                        help="Path to the CSV file containing predictions.")

    args = parser.parse_args()

    # Evaluate predictions
    evaluate_predictions(args.csv)
