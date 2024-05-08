import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keras

(values, labels), (values_2, labels_2) = keras.datasets.mnist.load_data()

values = np.append(values, values_2, axis=0)

for number in range(len(values)):
    number_formatted = np.array(values[number]).astype("uint8")
    plt.title(labels[number])
    plt.imshow(number_formatted, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.close()
