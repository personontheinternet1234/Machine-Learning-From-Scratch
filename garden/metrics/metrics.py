"""
result calculation functions
"""

import numpy as np


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
