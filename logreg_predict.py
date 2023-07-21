import numpy as np
import pandas as pd
import sys, os
from my_logistic_regression import MyLogisticRegression
from logreg_train import normalization

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} [predict path] [weight path]")
    else:
        data_predict = pd.read_csv(
            sys.argv[1]).set_index('Index')
        weights = pd.read_csv(sys.argv[2])

        label_map = {0: 'Ravenclaw', 1: 'Slytherin',
                     2: 'Gryffindor', 3: 'Hufflepuff'}

        data_predict = data_predict.drop(columns=['Arithmancy',
                                                  'Potions',
                                                  'Care of Magical Creatures'])
        data_predict = data_predict.drop('Hogwarts House', axis=1)
        print(data_predict.select_dtypes(include='number').columns)
        numeric_predict = data_predict.select_dtypes(include='number').values
        column_means = np.nanmean(numeric_predict, axis=0)
        # Fill missing values with column-wise averages
        for i in range(numeric_predict.shape[1]):
            column_mask = np.isnan(numeric_predict[:, i])
            numeric_predict[column_mask, i] = column_means[i]

        # numeric_predict = minmax(numeric_predict)
        numeric_predict = normalization(numeric_predict)
        # print(numeric_predict[0])

        num_classes = len(weights)

        result_array = np.array([])  # Initialize an empty array
        for i in range(num_classes):
            theta = weights.iloc[i].values.tolist()
            # theta_nested = [[value] for value in theta]
            # theta_nested.pop(0)
            # print(theta_nested)

            mylr = MyLogisticRegression(np.array(theta[1:]), 0.1, 1000, None, 0.0, 'batch')

            # print(len(numeric_predict))
            # # print(len(classzero_nested))

            y_pred = mylr.predict_(numeric_predict[0])

            y_pred_binary = (y_pred >= 0.5).astype(int)
            y_pred_binary = np.reshape(y_pred_binary, (-1, 1)).astype(str)

            for sublist in y_pred_binary:
                if int(sublist[0]) == 1:
                    sublist[0] = label_map[i]

            # filename = 'result_' + label_map[i] + '.csv'
            # np.savetxt(filename,
            #            y_pred_binary, delimiter=',', fmt='%s')

            # print(y_pred_binary)
            if result_array.size == 0:
                result_array = y_pred
            else:
                result_array = np.hstack((result_array, y_pred))

        print(result_array)
        # Define the label map
        label_map = {0: 'Ravenclaw', 1: 'Slytherin',
                     2: 'Gryffindor', 3: 'Hufflepuff'}

        # Process each row and find the highest value column
        result_df = pd.DataFrame(result_array)

        highest_labels = result_df.idxmax(axis=1).map(label_map)

        highest_labels = pd.DataFrame({"Hogwarts House": highest_labels})

        print(highest_labels)
        # Write the result to a new CSV file without the header
        highest_labels.to_csv('houses.csv', index_label='Index', header=True)

        # np.savetxt('result.csv', result_array, delimiter=',', fmt='%s')
