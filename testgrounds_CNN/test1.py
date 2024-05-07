import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from PIL import Image

# new image selection if wanted
# import random
# import keras
# (train_values, train_labels), (test_values, test_labels) = keras.datasets.mnist.load_data()
# test_image = np.array(train_values[random.randint(0, len(train_values))])
# print(test_image.tolist())

kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
test_image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 170, 255, 198, 29, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 141, 255, 255, 255, 255, 255, 226, 57, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 226, 255, 255, 198, 86, 86, 198, 255, 141, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 226, 255, 170, 29, 0, 0, 0, 170, 255, 198, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 226, 255, 86, 0, 0, 0, 0, 0, 141, 255, 255, 86, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 86, 255, 255, 86, 0, 0, 0, 0, 0, 0, 141, 255, 226, 29, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 57, 255, 255, 86, 0, 0, 0, 0, 0, 0, 29, 226, 255, 114, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 170, 255, 114, 0, 0, 0, 0, 0, 0, 0, 198, 255, 226, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 170, 255, 86, 0, 0, 0, 0, 0, 0, 226, 255, 255, 57, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 141, 255, 114, 0, 0, 0, 0, 141, 226, 255, 255, 114, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 198, 255, 198, 255, 255, 255, 255, 255, 255, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 198, 255, 255, 226, 226, 255, 255, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 198, 255, 114, 0, 0, 0, 0, 0, 29, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 141, 255, 198, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 255, 255, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 198, 255, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 198, 255, 226, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 255, 255, 255, 86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 255, 255, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 114, 255, 114, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
test_image = test_image.astype("uint8")
test_image_flat = test_image.flatten()

# Convert the test image to a PIL image
image = Image.open("off_center_test.jpeg")
image = image.convert("L")
image = list(image.getdata())
image2 = []
for i in range(28):
    image2.append(image[i*28:i*28+28])

image2 = np.array(image2).astype("uint8")
print(image2)

# Display the image using Matplotlib
plt.imshow(image2, cmap='gray')
plt.axis('off')
plt.show()

for i in range(14):
    convolved_image = convolve2d(image2, kernel, mode='valid')
    image2 = convolved_image
    plt.imshow(convolved_image, cmap='gray')
    plt.axis('off')
    plt.show()
