import numpy as np
import pandas as pd

test = pd.read_csv("saved/data_labels_keras.csv")
print(test)
for i in range(len(test)):
    test[i] = np.array([test[i]])
print(np.array(test).tolist())
