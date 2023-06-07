import sys
import pandas as pd

# https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas


def describe_dataset(filename):
    try:
        df = pd.read_csv(filename)
        numeric_features = df.select_dtypes(include='number')

        print("OG Data:", df.shape[1],
              "numeric only:", numeric_features.shape[1])
        if numeric_features.empty:
            print("No numerical features found in the dataset.")
            return

        description = numeric_features.describe()
        print(description)

    except FileNotFoundError:
        print("File not found. Please provide a valid filename.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py [dataset_filename]")
    else:
        dataset_filename = sys.argv[1]
        describe_dataset(dataset_filename)
