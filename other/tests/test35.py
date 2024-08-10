import random

import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import convolve2d

import keras

show_convolved = True
kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

(train_values, train_labels), (test_values, test_labels) = keras.datasets.mnist.load_data()
values = np.append(train_values, test_values, axis=0)
labels = np.append(train_labels, test_labels, axis=0)
while True:
    tc = random.randint(0, len(values))
    test_image = np.array(values[tc])
    print(test_image.tolist())

    test_image = test_image.astype("uint8")
    test_image_flat = test_image.flatten()

    # Display the image using Matplotlib
    plt.imshow(test_image, cmap="gray")
    plt.axis("off")
    plt.title(str(labels[tc]))
    plt.show()
    plt.close()

    if show_convolved:
        for i in range(14):
            convolved_image = convolve2d(test_image, kernel, mode="same")
            test_image = convolved_image
            plt.imshow(convolved_image, cmap="gray")
            plt.axis('off')
            plt.show()
