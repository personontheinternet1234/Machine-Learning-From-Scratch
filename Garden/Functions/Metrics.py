"""
Visual functions
Authors:
    Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>
    Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

cmap_green_d = sns.color_palette("dark:#2ecc71", as_cmap=True)
rev_cmap_green_d = cmap_green_d.reversed()


def generate_cm(label, predicted, normalize=True):
    """ generate a confusion matrix based on predictions """
    # find errors
    if len(label) != len(predicted):
        raise ValueError(f"'({len(label)})' is not the same as '{len(predicted)}'")

    # instantiate confusion matrix
    cm = []
    for i in range(len(set(label))):
        cm.append([0] * len(set(predicted)))

    # generate confusion matrix
    for i in range(len(label)):
        cm[label[i] - 1][predicted[i] - 1] += 1

    # normalize confusion matrix
    if normalize:
        for i in range(len(cm)):
            row_sum = np.sum(cm[i])
            if row_sum != 0:
                cm[i] = cm[i] / row_sum
            else:
                cm[i] = cm[i]

    # reformat and return confusion matrix
    cm = np.array(cm)
    return cm


def loss_graph(x, y, title=None, x_label=None, y_label=None):
    data = pd.DataFrame(x, y)
    sns.lineplot(data, x=x_label, y=y_label)
    plt.title(title)
    plt.show()


def prob_visual_cm(cm, title=None, x_labels=None, y_labels=None, x_axis='Predicted', y_axis='True', annotation=True):
    """ visual confusion matrix from probabilities """
    if not x_labels:
        x_labels = [i for i in range(0, len(cm))]
    if not y_labels:
        y_labels = [i for i in range(0, len(cm[0]))]
    ax = sns.heatmap(cm, annot=annotation, xticklabels=x_labels, yticklabels=y_labels, linewidth=.5, cmap='Greens', vmin=0, vmax=1)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(title)
    plt.show()
