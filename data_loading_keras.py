import numpy as np
import keras

# load data from mnist
(train_values, train_labels), (test_values, test_labels) = keras.datasets.mnist.load_data()
# combine training and testing labels
import_data_values = np.append(train_values, test_values, axis=0)
import_data_labels = np.append(train_labels, test_labels, axis=0)
data_values = []
data_labels = []

# reformat data from mnist and add labels
for i in range(len(train_values)):
    data_values.append(np.array([np.divide(import_data_values[i].flatten().tolist(), 255)]))
    node_values = np.zeros(10)
    node_values[train_labels[i]] = 1
    node_values = np.array([node_values])
    data_labels.append(node_values)

# save keras data
with open("saved/data_values_keras.txt", "w") as f:
    for array in data_values:
        f.write(str(array.tolist()) + "\n")
with open("saved/data_labels_keras.txt", "w") as f:
    for array in data_labels:
        f.write(str(array.tolist()) + "\n")
