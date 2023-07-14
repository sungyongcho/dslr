import pandas as pd
import matplotlib.pyplot as plt
import math


def histogram(numeric_features, houses):
    # Step 2: Calculate average scores for each house
    avg_scores = {}
    for house in houses:
        avg_scores[house] = {}

    # Step 3: Calculate average scores for each feature and house
    features = data.select_dtypes(include='number').columns
    # features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination',
    #             'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
    #             'Care of Magical Creatures', 'Charms', 'Flying']

    for feature in numeric_features:
        for house in houses:
            avg_scores[house][feature] = data[data['Hogwarts House']
                                              == house][feature]

    # Step 4: Create subplots for all features
    num_features = len(numeric_features)
    num_rows = num_features // 2 + num_features % 2
    num_cols = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 18))
    fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots

    for i, feature in enumerate(numeric_features):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]

        for house in houses:
            ax.hist(avg_scores[house][feature],
                    bins=20, alpha=0.3, label=house)

        ax.set_title(feature)
        ax.set_ylabel('Count')
        ax.legend(loc='upper right')

    # Hide empty subplots if the number of features is odd
    if num_features % 2 != 0:
        axes[-1, -1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('./datasets/dataset_train.csv').set_index('Index')
    numeric_features = data.select_dtypes(include='number').columns
    houses = data['Hogwarts House'].unique()
    histogram(numeric_features, houses)
