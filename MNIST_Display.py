import random
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import keras


# update image
def update(frame):
    global ax
    ax.clear()
    tc = random.randint(0, display_length - 1)
    ax.imshow(display_values[tc], cmap="gray")
    ax.set_title(display_labels[tc])
    ax.axis("off")


# params
frames = 10000
time_interval = 150
number = "random"
val_numbers = list("0123456789")

# load
(values, labels), (values_2, labels_2) = keras.datasets.mnist.load_data()

# combine train & test
values = np.append(values, values_2, axis=0)
labels = np.append(labels, labels_2, axis=0)

display_values = []
display_labels = []

if number == "random":
    display_values = values
    display_labels = labels
elif number in val_numbers:
    for i in range(len(labels)):
        if str(labels[i]) == number:
            display_values.append(values[i])
            display_labels.append(labels[i])
    display_values = np.array(display_values)
    display_labels = np.array(display_labels)
else:
    print("Invalid number")
    sys.exit(1)
display_length = len(display_values)

# instantiate plot
fig, ax = plt.subplots()
ax.set_title(f"{number} values of MNIST")
ax.axis("off")

# animate plot
ani = FuncAnimation(fig, update, frames=frames, interval=time_interval)
plt.show()
