import numpy as np
import pandas as pd

# test = np.array(pd.read_csv("saved/data_labels_keras.csv")).tolist()
# print(test)
# for i in range(len(test)):
#     test[i] = np.array([test[i]])
# print(test)


print("a")
test_2 = pd.read_csv("saved/data_labels_keras.csv", nrows=0).columns.tolist()

print(test_2)