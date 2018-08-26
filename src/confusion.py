

from __future__ import print_function
from __future__ import division
import sys
import pickle
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def confusion(file_path):
    labels = {
        'ADVE': 0,
        'Email': 1,
        'Form': 2,
        'Letter': 3,
        'Memo': 4,
        'News': 5,
        'Note': 6,
        'Report': 7,
        'Resume': 8,
        'Scientific': 9}

    reverse_labels = {v: k for k, v in labels.iteritems()}
    all_labels = [reverse_labels[x] for x in range(10)],

    with open(file_path, 'rb') as this:
        matrix = pickle.load(this)
        df_cm = pd.DataFrame(matrix, index=[i for i in all_labels],
                             columns=[i for i in all_labels])
        plt.figure(figsize=(10,7))
        sn.heatmap(df_cm, annot=True, cmap='YlGnBu')
        plt.show()


if __name__ == '__main__':
    confusion(file_path=sys.argv[1])
