import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import train_test_split
from my_logistic_regression import MyLogisticRegression as MyLR
from hogwarts_mapping import mapping, colors
from utils import load_data, normalization


def train(data, target_categories, param='batch'):
    weights = []
    x = data[:, :-1]
    y = data[:, -1]

    for house in target_categories:
        house_label = mapping[house]
        print(f"Current house: {house} (Label: {house_label})")

        # Convert the target data to binary labels for the current house
        y_labeled = np.where(y == house_label, 1, 0).reshape(-1, 1)
        classifier = MyLR(np.random.rand(
            x.shape[1] + 1, 1), 1e-1, 1000, None, 0.0, param)
        if (param == 'batch'):
            theta, accuracy, accuracy_list, loss_list, epoch_list = classifier.fit_(
                x, y_labeled)
        elif (param == 'sgd'):
            theta, accuracy, accuracy_list, loss_list, epoch_list = classifier.stochastic_fit(
                x, y_labeled)
        elif (param == 'mini'):
            theta, accuracy, accuracy_list, loss_list, epoch_list = classifier.mini_batch_fit(
                x, y_labeled)
        print('Accuracy: ', accuracy)

        # Save the weights for the current class
        weights.append(theta)
    return weights


def compare_optimization_algorithms(data, target_categories):
    x = data[:, :-1]
    y = data[:, -1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for house in target_categories:
        house_label = mapping[house]
        print(f"Current house: {house} (Label: {house_label})")

        y_labeled = np.where(y == house_label, 1, 0).reshape(-1, 1)

        # class
        classifier_batch = MyLR(np.random.rand(
            x.shape[1] + 1, 1), 1e-1, 1000, None, 0.0, 'batch')
        theta_batch, accuracy_batch, accuracy_list_batch, loss_list_batch, epoch_list_batch = classifier_batch.fit_(
            x, y_labeled)

        classifier_sgd = MyLR(np.random.rand(
            x.shape[1] + 1, 1), 1e-1, 1000, None, 0.0, 'sgd')
        theta_sgd, accuracy_sgd, accuracy_list_sgd, loss_list_sgd, epoch_list_sgd = classifier_sgd.stochastic_fit(
            x, y_labeled)

        classifier_mini = MyLR(np.random.rand(
            x.shape[1] + 1, 1), 1e-1, 1000, None, 0.0, 'mini')
        theta_mini, accuracy_mini, accuracy_list_mini, loss_list_mini, epoch_list_mini = classifier_mini.mini_batch_fit(
            x, y_labeled)

        print('Accuracy: ', accuracy_sgd, accuracy_mini, accuracy_batch)

        house_color = colors[house]

        axes[0].plot(epoch_list_sgd, loss_list_sgd,
                     label=house, color=house_color)
        axes[1].plot(epoch_list_mini, loss_list_mini,
                     label=house, color=house_color)
        axes[2].plot(epoch_list_batch, loss_list_batch,
                     label=house, color=house_color)

    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    # axes[0].set_ylim([0, 1])
    axes[0].set_title('[Stochastic] Loss vs Epoch by Hogwarts House')
    axes[0].legend()

    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')
    # axes[1].set_ylim([0, 1])
    axes[1].set_title('[Mini Batch] Loss vs Epoch by Hogwarts House')
    axes[1].legend()

    axes[2].set_xlabel('epoch')
    axes[2].set_ylabel('loss')
    # axes[2].set_ylim([0, 1])
    axes[2].set_title('[Batch] Loss vs Epoch by Hogwarts House')
    axes[2].legend()
    plt.tight_layout()
    plt.show()


def save(data, target_categories, params):
    print(f"Starting training each classifier for logistic regression...")
    weights = train(data, target_categories, params)

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Transpose the theta arrays
    transposed_thetas = [theta.T for theta in weights]

    # Concatenate the transposed thetas vertically
    all_thetas = np.concatenate(transposed_thetas, axis=0)

    # Create an array of class numbers
    class_numbers = np.arange(4).reshape(-1, 1)

    # Create a DataFrame to store the class numbers and theta values
    df = pd.DataFrame(np.concatenate((class_numbers, all_thetas), axis=1), columns=[
                      'Class'] + ['Theta_' + str(i) for i in range(data[:, :-1].shape[1] + 1)])

    # Save the DataFrame to a CSV file
    df.to_csv('weights.csv', index=False)


if __name__ == "__main__":
    if not (2 <= len(sys.argv) and len(sys.argv) <= 3):
        print(
            f"Usage:  python {sys.argv[0]} [data path] (for batch gradient descent)")
        print(f"\tpython {sys.argv[0]} [data path] [batch option]")
        print(
            f"\tthree batch options: batch, sgd (for stochastic), mini (for mini-batch)")
    else:
        # Load the data
        df, features = load_data(sys.argv[1])
        df = df.drop(columns=['Arithmancy', 'Potions',
                              'Care of Magical Creatures'])
        df.dropna(inplace=True)
        target_categories = df['Hogwarts House'].unique()

        x = df.select_dtypes(include='number')
        normalized_x, data_min, data_max = normalization(x.values)
        y = df['Hogwarts House'].replace(mapping).values
        new_data = np.column_stack((normalized_x, y))

        if len(sys.argv) == 2:
            save(new_data, target_categories, 'batch')
        else:
            save(new_data, target_categories, sys.argv[2])

        # Bonus: plotting to compare all three gradient descent methods
        print("\nComparing optimizaiton algoritms:")
        compare_optimization_algorithms(new_data, target_categories)
