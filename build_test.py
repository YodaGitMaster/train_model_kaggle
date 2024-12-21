import os
import pandas as pd
import argparse


def create_test_data(labels_path, sample_size, output_folder):
    """
    Create a test set by randomly sampling and shuffling from labels.csv.

    Args:
        labels_path (str): Path to labels.csv file.
        sample_size (int): Number of samples for the test set.
        output_folder (str): Folder to save the test data.
    """
    # Read the labels CSV
    labels_df = pd.read_csv(labels_path)

    # Sample the data (randomly pick and shuffle)
    sampled_df = labels_df.sample(
        n=sample_size, random_state=123).reset_index(drop=True)
    sampled_df = sampled_df.sample(
        frac=1, random_state=456).reset_index(drop=True)  # Shuffle rows

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the test data
    test_data_path = os.path.join(output_folder, f"testdata_{sample_size}.csv")
    sampled_df.to_csv(test_data_path, index=False)
    print(f"Test data saved to {test_data_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create test set from labels.csv.")
    parser.add_argument("-l", "--labels_path", type=str,
                        required=True, help="Path to labels.csv file.")
    parser.add_argument("-s", "--sample_size", type=int,
                        required=True, help="Number of samples for the test set.")
    parser.add_argument("-o", "--output_folder", type=str,
                        default="tests", help="Folder to save the test set.")
    args = parser.parse_args()

    # Create the test data
    create_test_data(args.labels_path, args.sample_size, args.output_folder)


if __name__ == "__main__":
    main()
