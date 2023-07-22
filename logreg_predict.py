import numpy as np
import pandas as pd
import sys
import os
from my_logistic_regression import MyLogisticRegression
from hogwarts_mapping import mapping
from utils import normalization, load_data, fill_columns_mean


def save_individual_results(y_pred):
    # if you want to check individual predicted values
    y_pred_binary = (y_pred >= 0.5).astype(int)
    y_pred_binary = np.reshape(y_pred_binary, (-1, 1)).astype(str)
    for sublist in y_pred_binary:
        if int(sublist[0]) == 1:
            house_name = list(mapping.keys())[
                list(mapping.values()).index(i)]
            sublist[0] = house_name

    filename = 'result_' + \
        list(mapping.keys())[list(mapping.values()).index(i)] + '.csv'
    np.savetxt(filename,
               y_pred_binary, delimiter=',', fmt='%s')
    # print(y_pred_binary)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} [predict path] [weights path]")
    else:
        data_predict, features = load_data(sys.argv[1])
        weights = pd.read_csv(sys.argv[2])

        data_predict = data_predict.drop(columns=['Arithmancy',
                                                  'Potions',
                                                  'Care of Magical Creatures',
                                                  'Hogwarts House'])

        numeric_predict = data_predict.select_dtypes(include='number').values

        numeric_predict = fill_columns_mean(numeric_predict)

        numeric_predict = normalization(numeric_predict)

        num_classes = len(weights)

        result_array = np.array([])  # Initialize an empty array
        for i in range(num_classes):
            theta = weights.iloc[i].values.tolist()

            # creating MyLogisticRegression object
            mylr = MyLogisticRegression(
                np.array(theta[1:]), 0.1, 1000, None, 0.0)

            y_pred = mylr.predict_(numeric_predict[0])

            # if you want to check predicted values
            # for individual models for each classes
            # save_individual_results(y_pred)

            if result_array.size == 0:
                result_array = y_pred
            else:
                result_array = np.hstack((result_array, y_pred))

        # if you want to see the predicted value
        # print(result_array)

        # Process each row and find the highest value column
        result_df = pd.DataFrame(result_array)

        highest_labels = result_df.idxmax(axis=1).map(
            {i: house_name for house_name, i in mapping.items()})

        result_mapped = pd.DataFrame({"Hogwarts House": highest_labels})

        # Write the result to a new CSV file without the header
        result_mapped.to_csv('houses.csv', index_label='Index', header=True)

        # to print
        print(result_mapped)
