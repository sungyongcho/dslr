import numpy as np
import pandas as pd
import os
import sys

from my_logistic_regression import MyLogisticRegression


def load_data(file_path):
    """
    Load the dataset from a CSV file.
    Assumes the target variable is in the last column.
    """
    data = pd.read_csv(file_path).set_index('Index')
    print(data.head)
    X = data.select_dtypes(include='number').dropna().values
    y = data['Hogwarts House'].values.reshape(-1, 1)
    print(X, y)
    return X, y


def save_weights(weights, file_path):
    """
    Save the weights to a file.
    """
    np.savetxt(file_path, weights, delimiter=',')


def logreg_train(dataset_path):
    """
    Train the logistic regression models using one-vs-all strategy.
    Save the weights to a file for each class.
    """
    X, y = load_data(dataset_path)

    # Normalize the features if needed
    # ...

    num_classes = np.unique(y).shape[0]
    weights = []

    for i in range(num_classes):
        # Create a binary target variable for the current class
        binary_target = (y == i).astype(int)
        binary_target = np.reshape(binary_target, (-1, 1))

        # Create an instance of MyLogisticRegression
        logistic_regression = MyLogisticRegression(
            theta=np.random.rand(X.shape[1] + 1, 1), alpha=1e-2, max_iter=100000)

        # Train the logistic regression model
        theta = logistic_regression.fit_(X, binary_target)

        # Save the weights for the current class
        weights.append(theta)

        print("Class", i)
        print("Theta:", theta.flatten())

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Save the weights to separate files for each class
    for i, theta in enumerate(weights):
        weights_path = os.path.join(script_dir, f"weights_class_{i}.csv")
        save_weights(theta, weights_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    # load_data(dataset_path)

    logreg_train(dataset_path)
