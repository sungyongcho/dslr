import sys
import pandas as pd

# https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas


def describe_dataset(filename):
    try:
        df = pd.read_csv(filename).set_index('Index')
        numeric_features = df.select_dtypes(include='number')
        summary_df = pd.DataFrame()
        for column in numeric_features.columns:
            values = df[column]
            values = values.dropna()
            # Calculate statistics manually
            count = len(values)
            mean = sum(values) / count
            stddev = (sum((x - mean) ** 2 for x in values) / count) ** 0.5
            minimum = min(values)
            quartiles = [values.quantile(q) for q in [0.25, 0.50, 0.75]]
            maximum = max(values)

            # Create a temporary DataFrame with custom index
            index_labels = ['Count', 'Mean', 'Std',
                            'Min', '25%', '50%', '75%', 'Max']
            temp_df = pd.DataFrame([count, mean, stddev, minimum, quartiles[0],
                                    quartiles[1], quartiles[2], maximum],
                                   index=index_labels, columns=[column])

            # Append the temporary DataFrame to the summary DataFrame
            summary_df = pd.concat([summary_df, temp_df], axis=1)

        print(summary_df)
        # to compare result with pandas describe funciton
        # print(numeric_features.describe())
    except FileNotFoundError:
        print("File not found. Please provide a valid filename.")


if len(sys.argv) != 2:
    print("Usage: python describe.py [dataset_filename]")
else:
    dataset_filename = sys.argv[1]
    describe_dataset(dataset_filename)
