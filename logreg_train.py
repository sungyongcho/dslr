import numpy as np
import pandas as pd
import os
import sys
from minmax import minmax

from my_logistic_regression import MyLogisticRegression

label_map = {0: 'Ravenclaw', 1: 'Slytherin',
             2: 'Gryffindor', 3: 'Hufflepuff'}


def load_data(file_path):
    """
    Load the dataset from a CSV file.
    Assumes the target variable is in the last column.
    """
    reverse_label_map = {v: k for k, v in label_map.items()}

    data = pd.read_csv(file_path).set_index('Index')
    # print(data.head)
    data = data.drop(columns=['Arithmancy',
                              'Potions',
                              'Care of Magical Creatures'])
    data = data.dropna()
    X = data.select_dtypes(include='number').values
    y = data['Hogwarts House']
    y = y.map(label_map)
    y = y.values.reshape(-1, 1)
    print(X.shape, y.shape)

    # print(X, y)
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

    weights = []

    X = minmax(X)

    # # y_numeric = [class_names.index(name[0]) for name in y]
    for i in range(4):
        # Create a binary target variable for the current class
        binary_target = (y == i).astype(int)
        binary_target = np.reshape(binary_target, (-1, 1))
        # if i == 0:
        #     np.savetxt('griff_binary.csv', binary_target, delimiter=',')
        # print(binary_target)

        # print(X.shape)
        # Create an instance of MyLogisticRegression and train
        logistic_regression = MyLogisticRegression(
            theta=np.random.rand(X.shape[1] + 1, 1), alpha=1e-1, max_iter=1000)
        theta = logistic_regression.fit_(X, binary_target)

        # stochastic gradient descent
        # logistic_regression = MyLogisticRegression(
        #     theta=np.random.rand(X.shape[1] + 1, 1), alpha=1e-2, max_iter=100000)
        # theta = logistic_regression.stochastic_fit_(X, binary_target)

        # Save the weights for the current class
        weights.append(theta)
        # print(weights)

        print("Class", i)
        print("Theta:", theta)

    # Transpose the theta arrays
    transposed_thetas = [theta.T for theta in weights]

    # Concatenate the transposed thetas vertically
    all_thetas = np.concatenate(transposed_thetas, axis=0)

    # Calculate the number of classes based on the label_map
    num_classes = len(label_map)

    # Create an array of class numbers
    class_numbers = np.arange(num_classes).reshape(-1, 1)

    # Create a DataFrame to store the class numbers and theta values
    df = pd.DataFrame(np.concatenate((class_numbers, all_thetas), axis=1), columns=[
                      'Class'] + ['Theta_' + str(i) for i in range(X.shape[1] + 1)])

    # Save the DataFrame to a CSV file
    df.to_csv('weights.csv', index=False)

    # # Save the weights to separate files for each class
    # for i, theta in enumerate(weights):
    #     weights_path = os.path.join(script_dir, f"weights_class_{i}.csv")
    #     save_weights(theta, weights_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    # load_data(dataset_path)

    logreg_train(dataset_path)
