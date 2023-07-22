import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
import sys
from hogwarts_mapping import colors


def pair(data):
    # Step 3: Create pair plot or scatter plot matrix for categorical features
    houses = data['Hogwarts House'].unique()

    # Increase the size of the figure and adjust padding/margins
    sns.set(font_scale=1.0)
    plt.figure(figsize=(16, 12))  # Increase the size as desired
    # Adjust the size of the dots
    plot_kws = {'s': 15}  # Modify the size as desired

    # Rotate the y-labels
    g = sns.pairplot(data, kind='scatter', diag_kind='hist',
                     height=1.5, aspect=1.5, hue='Hogwarts House',
                     palette=colors, plot_kws=plot_kws)

    for ax in g.axes.flat:
        ax.set_ylabel(ax.get_ylabel(), rotation=45,
                      horizontalalignment='right')
        ax.set_xlabel(ax.get_xlabel(), rotation=45,
                      horizontalalignment='right')

    # Adjust padding/margins for y-axis labels
    g.fig.subplots_adjust(left=0.08, bottom=0.12)

    # Save the pair plot as an image
    plt.savefig('pair.png', dpi=300)

    # if you want to see
    # plt.show()

    print('done')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py [dataset_filename]")
    else:
        data = pd.read_csv(sys.argv[1]).set_index('Index')
        # print(data.columns)
        numeric_features = data.select_dtypes(include='number').columns
        numeric_features = numeric_features.append(
            pd.Index([data['Hogwarts House']]))
        pair(data)
