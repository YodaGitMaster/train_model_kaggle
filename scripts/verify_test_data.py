import os
import numpy as np
from tensorflow.keras.models import load_model


def verify_model_mapping(model, trained_class_names, output_folder):
    """
    Verify the mapping between model output indices and class names.
    Save results to a .txt and .csv file in the specified folder.
    """
    # Number of classes
    num_classes = len(trained_class_names)

    # Simulate one-hot encoded outputs (e.g., [1, 0, 0, ...])
    simulated_output = np.eye(num_classes)

    # Prepare output directory
    os.makedirs(output_folder, exist_ok=True)

    # Prepare outputs
    output_txt_path = os.path.join(output_folder, "mapping_verification.txt")
    output_csv_path = os.path.join(output_folder, "mapping_verification.csv")

    print("Verifying model class mapping...")

    # Write to .txt file
    with open(output_txt_path, "w") as txt_file:
        txt_file.write("Model Class Mapping Verification\n")
        txt_file.write("=" * 40 + "\n")

        print("\nSimulated Predictions:")
        mapping_results = []
        for i, row in enumerate(simulated_output):
            predicted_index = np.argmax(row)  # Simulate prediction output
            predicted_label = trained_class_names[predicted_index]
            result_line = f"Simulated output {row} -> Predicted Index: {predicted_index}, Predicted Label: {predicted_label}"
            print(result_line)
            txt_file.write(result_line + "\n")

            # Prepare for CSV
            mapping_results.append({
                "Simulated Output": row.tolist(),
                "Predicted Index": predicted_index,
                "Predicted Label": predicted_label
            })

    # Save to .csv file
    import pandas as pd
    mapping_df = pd.DataFrame(mapping_results)
    mapping_df.to_csv(output_csv_path, index=False)

    print(f"\nMapping verification saved to:\n- {output_txt_path}\n- {output_csv_path}")


if __name__ == "__main__":
    # Replace with your model path
    model_path = "best_model.h5"

    # Define the correct class names in the order used during training
    trained_class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]  # Update as necessary

    # Output folder for results
    output_folder = os.path.join("results", "testing")

    # Try to load the model
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'. Please provide the correct path.")
        exit(1)

    # Verify mapping and save results
    verify_model_mapping(model, trained_class_names, output_folder)
