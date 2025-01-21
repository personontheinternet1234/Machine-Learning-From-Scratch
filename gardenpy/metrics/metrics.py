r"""
**GardenPy raw data interpretation tools.**

Contains:
    - :func:`confusion_matrix`
"""

import numpy as np


def confusion_matrix(label, predicted, *, normalize=True) -> np.ndarray:
    r"""
    ...
    """
    # errors
    if len(label) != len(predicted):
        raise ValueError(f"'({len(label)})' is not the same as '{len(predicted)}'")

    # instantiate cm
    matrix = []
    for i in range(len(set(label))):
        matrix.append([0] * len(set(predicted)))

    # generate cm
    for i in range(len(label)):
        matrix[label[i] - 1][predicted[i] - 1] += 1

    # normalize cm
    if normalize:
        for i in range(len(matrix)):
            row_sum = np.sum(matrix[i])
            if row_sum != 0:
                matrix[i] = matrix[i] / row_sum
            else:
                matrix[i] = matrix[i]

    # return cm
    matrix = np.array(matrix)
    return matrix
