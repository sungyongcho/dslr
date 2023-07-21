import pandas as pd
import numpy as np

# Replace "file1.csv", "file2.csv", "file3.csv", and "file4.csv" with your actual file paths.
file1 = pd.read_csv("result_Gryffindor.csv", header=None, squeeze=True)
file2 = pd.read_csv("result_Hufflepuff.csv", header=None, squeeze=True)
file3 = pd.read_csv("result_Ravenclaw.csv", header=None, squeeze=True)
file4 = pd.read_csv("result_Slytherin.csv", header=None, squeeze=True)
# Combine the data based on the condition you provided
combined_data = file4.where(file3 == '0', file3).where(
    file2 == '0', file2).where(file1 == '0', file1)

# Write the combined data to a new CSV file
combined_data.to_csv('merged_data.csv', index=False, header=False)
