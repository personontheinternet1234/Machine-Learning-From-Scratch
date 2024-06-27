"""
Data generator for checkered and non-checkered data
"""

import os
import shutil

import pandas as pd

# file paths
root = os.path.join(os.path.dirname(__file__), 'data')

# clear root
shutil.rmtree(root)
os.makedirs(root)

# create data lists
values = []
labels = []

# generate data
for i in range(4):
    values.append([int(i > 1), int(i % 2 != 0)])
    labels.append([int(values[i][0] != values[i][1]), int(values[i][0] == values[i][1])])

# make data dataframe
values = pd.DataFrame(values, columns=['pixel-1', 'pixel-2'])
labels = pd.DataFrame(labels, columns=['checkered', 'non-checkered'])

# save data
values.to_csv(os.path.join(root, 'values.csv'), index=False)
labels.to_csv(os.path.join(root, 'labels.csv'), index=False)
