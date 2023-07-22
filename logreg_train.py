import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import train_test_split
from my_logistic_regression import MyLogisticRegression as MyLR


def load_data(path):
    print(f"path: {path}")
    try:
        df = pd.read_csv(path, index_col=0)
    except:
        print("Invalid file error.")
        sys.exit()
    print("df shape:", df.shape)
    features = df.columns.tolist()

    return (df, features)


def normalization(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data, data_min, data_max


def denormalization(normalized_data, data_min, data_max):
    x = normalized_data * (data_max - data_min)
    denormalized_data = normalized_data * (data_max - data_min) + data_min
    return denormalized_data


def denormalize_thetas(thetas, data_max, data_min):
    # Recover the slope of the line
    slope = thetas[1] * (data_max[1] - data_min[1]) / \
        (data_max[0] - data_min[0])
    # Recover the intercept of the line
    intercept = thetas[0] * (data_max[1] - data_min[1]) + \
        data_min[1] - slope * data_min[0]
    denormalized_thetas = np.array([intercept, slope]).reshape(-1, 1)
    return denormalized_thetas


def label_data(y, house):
    y_ = np.zeros(y.shape)
    y_[np.where(y == int(house))] = 1
    y_labelled = y_.reshape(-1, 1)
    # print("y_labelled shape:", y_labelled.shape)
    # print("y_labelled[:5]:", y_labelled[:5])
    return y_labelled


def data_spliter_by(x, y, house):
    # print("y:", y, "house:", house)
    y_ = np.zeros(y.shape)
    y_[np.where(y == (house))] = 1
    y_labelled = y_.reshape(-1, 1)
    # print("y_labelled shape:", y_labelled.shape)
    # print("y_labelled[:5]:", y_labelled[:5])
    return train_test_split(x, y_labelled, test_size=0.2, random_state=42)


def predict_(x, thetas):
    for v in [x, thetas]:
        if not isinstance(v, np.ndarray):
            print(f"Invalid input: argument {v} of ndarray type required")
            return None

    if not x.ndim == 2:
        print("Invalid input: wrong shape of x", x.shape)
        return None

    if thetas.ndim == 1 and thetas.size == x.shape[1] + 1:
        thetas = thetas.reshape(x.shape[1] + 1, 1)
    elif not (thetas.ndim == 2 and thetas.shape == (x.shape[1] + 1, 1)):
        print(f"p Invalid input: wrong shape of {thetas}", thetas.shape)
        return None

    X = np.hstack((np.ones((x.shape[0], 1)), x))
    return np.array(1 / (1 + np.exp(-X.dot(thetas))))


def train(data, target_categories, param='batch'):
    weights = []
    x = data[:, :-1]
    for house in range(len(target_categories)):
        print(f"Current house: {house}")
        y_labelled = label_data(data[:, -1], house)
        classifier = MyLR(np.random.rand(
            x.shape[1] + 1, 1), 1e-1, 1000, None, 0.0, param)
        if (param == 'batch'):
            theta, accuracy, accuracy_list, loss_list, epoch_list = classifier.fit_(
                x, y_labelled)
        elif (param == 'sgd'):
            theta, accuracy, accuracy_list, loss_list, epoch_list = classifier.stochastic_fit(
                x, y_labelled)
        elif (param == 'mini'):
            theta, accuracy, accuracy_list, loss_list, epoch_list = classifier.mini_batch_fit(
                x, y_labelled)
        print('Accuracy: ', accuracy)

        # Save the weights for the current class
        weights.append(theta)
    return weights


def compare_optimization_algorithms(data, target_categories):
    x = data[:, :-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for house in range(len(target_categories)):
        print(f"Current house: {house}")
        y_labelled = label_data(data[:, -1], house)
        # theta = fit_(x, y_labelled, np.random.rand(x.shape[1] + 1, 1), 1e-1, 1000)

        # class
        classifier = MyLR(np.random.rand(
            x.shape[1] + 1, 1), 1e-1, 1000, None, 0.0, 'batch')
        theta, accuracy, accuracy_list, loss_list, epoch_list = classifier.fit_(
            x, y_labelled)
        classifier = MyLR(np.random.rand(
            x.shape[1] + 1, 1), 1e-1, 1000, None, 0.0, 'sgd')
        # classifier.optimizer = 'sgd'
        theta_sgd, accuracy_sgd, accuracy_list_sgd, loss_list_sgd, epoch_list_sgd = classifier.stochastic_fit(
            x, y_labelled)
        classifier = MyLR(np.random.rand(
            x.shape[1] + 1, 1), 1e-1, 1000, None, 0.0, 'mini')
        # classifier.optimizer = 'mini'
        theta_mini, accuracy_mini, accuracy_list_mini, loss_list_mini, epoch_list_mini = classifier.mini_batch_fit(
            x, y_labelled)

        print('Accuracy: ', accuracy_sgd, accuracy_mini, accuracy)
        axes[0].plot(epoch_list_sgd, loss_list_sgd,
                     label=target_categories[house])
        axes[1].plot(epoch_list_mini, loss_list_mini,
                     label=target_categories[house])
        axes[2].plot(epoch_list, loss_list, label=target_categories[house])

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


def save(data, target_categories):
    print(f"Starting training each classifier for logistic regression...")
    weights = train(data, target_categories, param='mini')

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
    # 그냥이면은 batch로
    # 세번째 argv가 들어오면 batch, sgd, mini 로 돌아가도록
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} [data path]")
    else:
        # Load the data
        df, features = load_data(sys.argv[1])
        df = df.drop(columns=['Arithmancy', 'Potions',
                              'Care of Magical Creatures'])
        df.dropna(inplace=True)
        target_categories = df['Hogwarts House'].unique()

        # Map unique values to numbers
        mapping = {'Ravenclaw': 0, 'Slytherin': 1,
                   'Gryffindor': 2, 'Hufflepuff': 3}

        x = df.select_dtypes(include='number')
        normalized_x, data_min, data_max = normalization(x.values)
        y = df['Hogwarts House'].replace(mapping).values
        new_data = np.column_stack((normalized_x, y))
        # print(target_categories)
        save(new_data, target_categories)
        # compare_optimization_algorithms(new_data, target_categories)
