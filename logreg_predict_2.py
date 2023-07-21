import pickle
import pandas as pd
import numpy as np
from minmax import minmax

from my_logistic_regression import MyLogisticRegression


label_map = {0: 'Ravenclaw', 1: 'Slytherin',
             2: 'Gryffindor', 3: 'Hufflepuff'}


def load_models():
    # Load the models from the pickle file
    filename = "models.pickle"
    with open(filename, 'rb') as file:
        models = pickle.load(file)

    classifiers = []
    # for house, classifier_data in models.items():
    #     # print(f"\nhouse:{house}")
    #     for classifier in classifier_data:
    #         # print(classifier)

    return models


data_predict = pd.read_csv(
    './datasets/dataset_test.csv').set_index('Index')
data_predict = data_predict.drop(columns=['Arithmancy', 'Potions',
                                          'Care of Magical Creatures'])
data_predict = data_predict.drop('Hogwarts House', axis=1)
numeric_predict = data_predict.select_dtypes(include='number').values
column_means = np.nanmean(numeric_predict, axis=0)
# Fill missing values with column-wise averages
for i in range(numeric_predict.shape[1]):
    column_mask = np.isnan(numeric_predict[:, i])
    numeric_predict[column_mask, i] = column_means[i]

numeric_predict = minmax(numeric_predict)


models = load_models()

print(len(models[0]))

print(models[0][6]['thetas'])

mylr = MyLogisticRegression(models[0][4]['thetas'])
y_pred = mylr.predict_(numeric_predict)
y_pred_binary = (y_pred >= 0.5).astype(int).flatten()
y_pred_binary = np.reshape(y_pred_binary, (-1, 1)).astype(str)

print(y_pred_binary)
