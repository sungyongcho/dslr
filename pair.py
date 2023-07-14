import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns


# what's ANOVA?? it's not used in here

def pair(data):
    # # Step 3: Create pair plot or scatter plot matrix for categorical features
    # Adjust the size according to your desired pixel values

    sns.pairplot(data, kind='scatter', diag_kind='hist',
                 height=0.8, aspect=1.2)
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('./datasets/dataset_train.csv').set_index('Index')
    numeric_features = data.select_dtypes(include='number').columns
    pair(data[numeric_features])
