import numpy as np
import pandas as pd
import os
from minmax import minmax
from my_logistic_regression import MyLogisticRegression

if __name__ == "__main__":
    data_predict = pd.read_csv(
        './datasets/dataset_test.csv').set_index('Index')
    weights = pd.read_csv('./weights.csv')

    label_map = {0: 'Gryffindor', 1: 'Hufflepuff',
                 2: 'Ravenclaw', 3: 'Slytherin'}

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

    num_classes = len(weights)

    result_array = np.array([])  # Initialize an empty array
    for i in range(num_classes):
        theta = weights.iloc[i].values.tolist()
        theta_nested = [[value] for value in theta]
        theta_nested.pop(0)
        # print(classzero_nested)

        mylr = MyLogisticRegression(theta_nested)

        # print(len(numeric_predict))
        # # print(len(classzero_nested))

        y_pred = mylr.predict_(numeric_predict)

        # y_pred_binary = (y_pred >= 0.5).astype(int).flatten()
        # y_pred_binary = np.reshape(y_pred_binary, (-1, 1)).astype(str)

        # for sublist in y_pred_binary:
        #     if int(sublist[0]) == 1:
        #         sublist[0] = label_map[i]

        # print(y_pred_binary)
        if result_array.size == 0:
            result_array = y_pred
        else:
            result_array = np.hstack((result_array, y_pred))

    print(result_array)
    np.savetxt('result.csv', result_array, delimiter=',', fmt='%s')
