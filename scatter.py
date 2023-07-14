import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns

def scatter(numeric_features):

    # Step 2: Calculate correlation matrix
    correlation_matrix = numeric_features.corr()

    print(correlation_matrix)

    # Step 3: Find similar features
    similar_features = []
    threshold = 0.8  # Set correlation threshold for similarity

    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) >= threshold:
                similar_features.append(
                    (correlation_matrix.columns[i], correlation_matrix.columns[j]))

    # Step 4: Create scatter plots for similar features
    for feature1, feature2 in similar_features:
        plt.scatter(data[feature1], data[feature2])
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f"Scatter plot: {feature1} vs {feature2}")
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv('./datasets/dataset_train.csv').set_index('Index')
    numeric_features = data.select_dtypes(include='number').columns
    scatter(data[numeric_features])
