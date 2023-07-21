import pandas as pd

# Read the data from the CSV file
df = pd.read_csv('result.csv', header=None)

# Define the label map
label_map = {0: 'Ravenclaw', 1: 'Slytherin',
             2: 'Gryffindor', 3: 'Hufflepuff'}

# Process each row and find the highest value column
highest_labels = df.idxmax(axis=1).map(label_map)

# Write the result to a new CSV file without the header
highest_labels.to_csv('merged_data.csv', index=False, header=False)
