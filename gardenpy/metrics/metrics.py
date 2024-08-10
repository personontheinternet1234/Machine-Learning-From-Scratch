r"""
'metrics' includes calculations for data evaluation for GardenPy.

'metrics' includes:
    'cm': A confusion matrix generator.

Refer to 'todo' for in-depth documentation on these metrics.
"""

import numpy as np


def cm(label, predicted, normalize=True):
    """ generate a confusion matrix based on predictions """
    # find errors
    if len(label) != len(predicted):
        raise ValueError(f"'({len(label)})' is not the same as '{len(predicted)}'")

    # instantiate confusion matrix
    matrix = []
    for i in range(len(set(label))):
        matrix.append([0] * len(set(predicted)))

    # generate confusion matrix
    for i in range(len(label)):
        matrix[label[i] - 1][predicted[i] - 1] += 1

    # normalize confusion matrix
    if normalize:
        for i in range(len(matrix)):
            row_sum = np.sum(matrix[i])
            if row_sum != 0:
                matrix[i] = matrix[i] / row_sum
            else:
                matrix[i] = matrix[i]

    # reformat and return confusion matrix
    matrix = np.array(matrix)
    return matrix
