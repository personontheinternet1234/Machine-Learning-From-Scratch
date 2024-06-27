"""
Data generator for checkered and non-checkered data
"""

import os

import pandas as pd

root = os.path.join(os.path.dirname(__file__), 'data')

values = []
labels = []

for i in range(4):
    values.append([int(i > 1), int(i % 2 != 0)])
    labels.append([int(values[i][0] != values[i][1]), int(values[i][0] == values[i][1])])

values = pd.DataFrame(values, columns=['pixel-1', 'pixel-2'])
labels = pd.DataFrame(labels, columns=['checkered', 'non-checkered'])

values.to_csv(os.path.join(root, 'values.csv'), index=False)
labels.to_csv(os.path.join(root, 'labels.csv'), index=False)
