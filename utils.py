import numpy as np
import pandas as pd
import sys


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
