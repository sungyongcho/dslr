import pandas as pd
import matplotlib.pyplot as plt


def scatter(numeric_features):

    # Step 2: Calculate correlation matrix
    correlation_matrix = numeric_features.corr()
    pd.set_option('display.max_columns', None)

    print(correlation_matrix)

    # Step 3: Find similar features
    similar_features = []
    threshold = 0.8  # Set correlation threshold for similarity

    feature1_header = "Feature 1"
    feature2_header = "Feature 2"
    threshold_header = "Threshold"

    # Get the maximum length of feature names for alignment
    max_feature_length = max(map(len, correlation_matrix.columns))

    # Print the headers
    print()
    print(f"{'Feature 1': <{max_feature_length}} {'Feature 2': <{max_feature_length}} Threshold")

    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) >= threshold:
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                similar_features.append(
                    (correlation_matrix.columns[i], correlation_matrix.columns[j]))
                correlation = abs(correlation_matrix.iloc[i, j])
                print(
                    f"{feature1: <{max_feature_length}} {feature2: <{max_feature_length}} {correlation:.16f}")

    # Step 4: Create scatter plots for similar features
    for feature1, feature2 in similar_features:
        plt.figure(figsize=(8, 6))  # Create a new figure for each scatter plot

        # Group data points by 'Hogwarts House'
        houses = data['Hogwarts House'].unique()
        colors = {'Ravenclaw': '#2b7bba', 'Slytherin': '#138b4a',
                  'Gryffindor': '#c72c41', 'Hufflepuff': '#e9c24d'}

        for house, color in zip(houses, colors):
            house_data = data[data['Hogwarts House'] == house]
            plt.scatter(house_data[feature1],
                        house_data[feature2], color=colors[house], label=house, s=25)

        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f"Scatter plot: {feature1} vs {feature2}")

        # Add legend for Hogwarts House labels and colors
        house_legend = [plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=colors[house], markersize=6, label=house)
                        for house, color in zip(houses, colors)]
        plt.legend(handles=house_legend)

        plt.show()


if __name__ == "__main__":
    data = pd.read_csv('./datasets/dataset_train.csv').set_index('Index')
    numeric_features = data.select_dtypes(include='number').columns
    scatter(data[numeric_features])
