import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import keras


# update image
def update(frame):
    global ax
    ax.clear()
    tc = random.randint(0, data_len - 1)
    ax.imshow(values[tc], cmap="gray")
    ax.set_title(labels[tc])
    ax.axis("off")


# params
frames = 10000
time_interval = 250

# load
(values, labels), (values_2, labels_2) = keras.datasets.mnist.load_data()

# combine train & test
values = np.append(values, values_2, axis=0)
labels = np.append(labels, labels_2, axis=0)
data_len = len(values)

# instantiate plot
fig, ax = plt.subplots()
ax.set_title("MNIST")
ax.axis("off")

# animate plot
ani = FuncAnimation(fig, update, frames=frames, interval=time_interval)
plt.show()
