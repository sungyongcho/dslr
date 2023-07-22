import sys
import pandas as pd

# https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas

# for all use


def get_non_missing_values(values):
    non_missing_values = []
    for value in values:
        if value != value:
            continue
        else:
            non_missing_values.append(value)
    return non_missing_values


def calculate_count(values, missing='y'):
    values = get_non_missing_values(values) if missing == 'y' else values
    count = 0
    for _ in values:
        count += 1
    return count

# for numerical data


def calculate_mean(values):
    count = calculate_count(values)
    mean = 0
    non_missing_values = get_non_missing_values(values)
    for element in non_missing_values:
        if isinstance(element, pd.Timestamp):
            tmp = (element - pd.Timestamp(0)).total_seconds()
            tmp = float(tmp)
            mean += tmp
        else:
            mean += element
    mean /= float(count)
    return mean


def calculate_stddev(values):
    squared_diff_sum = 0
    non_missing_values = get_non_missing_values(values)
    mean = calculate_mean(values)
    count = calculate_count(values)
    for value in non_missing_values:
        if value != value:
            continue
        else:
            squared_diff_sum += (value - mean) ** 2
    # Use (count - 1) for sample standard deviation
    stddev = (squared_diff_sum / (count - 1)) ** 0.5
    return stddev


def get_minimum(values):
    non_missing_values = get_non_missing_values(values)
    minimum = non_missing_values[0]
    for value in non_missing_values:
        if value < minimum:
            minimum = value
    return minimum


def sort_values(values):
    sorted_values = values.copy()
    n = len(sorted_values)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if sorted_values[j] > sorted_values[j + 1]:
                sorted_values[j], sorted_values[j +
                                                1] = sorted_values[j + 1], sorted_values[j]
    return sorted_values


def calculate_quartiles(values):
    non_missing_values = get_non_missing_values(values)
    all_values = sort_values(non_missing_values)
    n = calculate_count(all_values)

    q1_index = (n - 1) * 0.25
    q2_index = (n - 1) * 0.5
    q3_index = (n - 1) * 0.75

    # pandas uses linear_interpolation for calculating quartiles
    q1 = interpolate_value(all_values, q1_index)
    q2 = interpolate_value(all_values, q2_index)
    q3 = interpolate_value(all_values, q3_index)

    return [q1, q2, q3]


def interpolate_value(values, index):
    lower_index = int(index)
    upper_index = lower_index + 1
    weight = index - lower_index

    lower_value = values[lower_index]
    upper_value = values[upper_index] if upper_index < len(
        values) else values[lower_index]

    interpolated_value = lower_value + (upper_value - lower_value) * weight
    return interpolated_value


def get_maximum(values):
    non_missing_values = get_non_missing_values(values)
    maximum = non_missing_values[0]
    for value in non_missing_values:
        if value > maximum:
            maximum = value
    return maximum
# for numerical data

# for categorical data


def calculate_unique(values):
    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return len(unique_values)


def calculate_top(values):
    frequency = {}
    max_count = 0
    top_value = None
    for value in values:
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1
        if frequency[value] > max_count:
            max_count = frequency[value]
            top_value = value
    return top_value


def calculate_freq(values):
    frequency = {}
    for value in values:
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1
    max_count = 0
    freq_value = None
    for value, count in frequency.items():
        if count > max_count:
            max_count = count
            freq_value = value
    return max_count
# for categorical data


def describe_dataset(filename):
    try:
        df = pd.read_csv(filename, index_col='Index')

        # print(df.dtypes)

        df['Birthday'] = pd.to_datetime(df['Birthday'])

        df.drop(['First Name', 'Last Name'], axis=1, inplace=True)

        summary_df = pd.DataFrame()
        for column in df.columns:
            if df.dtypes[column] == 'float64':
                values = df[column]
                count = calculate_count(values)
                mean = calculate_mean(values)
                stddev = calculate_stddev(values)
                minimum = get_minimum(values)
                quartiles = calculate_quartiles(values)
                maximum = get_maximum(values)

                # Create a temporary DataFrame with custom index
                index_labels = ['Count', 'Mean', 'Std',
                                'Min', '25%', '50%', '75%', 'Max']
                temp_df = pd.DataFrame([count, mean, stddev, minimum, quartiles[0],
                                        quartiles[1], quartiles[2], maximum],
                                       index=index_labels, columns=[column])

            elif df.dtypes[column] == 'object':
                values = df[column]
                non_missing_values = get_non_missing_values(values)
                count = calculate_count(non_missing_values)
                unique = calculate_unique(non_missing_values)
                top = calculate_top(non_missing_values)
                freq = calculate_freq(non_missing_values)

                index_labels = ['Count', 'Unique', 'Top', 'Freq']
                temp_df = pd.DataFrame([count, unique, top, freq],
                                       index=index_labels, columns=[column])

            elif df.dtypes[column] == 'datetime64[ns]':
                values = df[column]
                count = calculate_count(values)
                mean = pd.Timestamp(calculate_mean(values), unit='s')
                minimum = get_minimum(values)
                quartiles = calculate_quartiles(values)
                maximum = get_maximum(values)

                index_labels = ['Count', 'Mean',
                                'Min', '25%', '50%', '75%', 'Max']
                temp_df = pd.DataFrame([count, mean, minimum, quartiles[0],
                                        quartiles[1], quartiles[2], maximum],
                                       index=index_labels, columns=[column])

            # Append the temporary DataFrame to the summary DataFrame
            summary_df = pd.concat([summary_df, temp_df], axis=1)

        # to show all the columns
        pd.set_option('display.max_columns', None)

        print(summary_df)

        # # to compare result with pandas describe funciton
        # print("================================")
        # print(df.describe(include='all', datetime_is_numeric=True))
    except FileNotFoundError:
        print("File not found. Please provide a valid filename.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py [dataset_filename]")
    else:
        dataset_filename = sys.argv[1]
        describe_dataset(dataset_filename)
