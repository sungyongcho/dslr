import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Read the merged data (predicted) and ground truth data from CSV files
merged_data = pd.read_csv('before.csv', header=0, index_col=0, squeeze=True)
ground_truth = pd.read_csv(
    './datasets/dataset_truth.csv', header=0, index_col=0, squeeze=True)

# Calculate accuracy
accuracy = classification_report(ground_truth, merged_data)

print("Accuracy:", accuracy)
