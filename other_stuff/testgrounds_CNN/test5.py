import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from PIL import Image

k1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # left edge
k2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # right edge
k3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # top
k4 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # bottom
kb = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256  # blur


img = Image.open(f'test_images/25-256245_m copy.jpg')
gray_img = img.convert('L')
test_image = np.array(gray_img)
test_image = test_image.astype('uint8')

plt.imshow(test_image, cmap='gray')
plt.title('1')
plt.axis('off')
plt.show()
plt.close()

for i in range(14):
    convolved_image = convolve2d(test_image, k1, mode='same')
    test_image = convolved_image
    plt.imshow(convolved_image, cmap='gray')
    plt.title(str(i + 1))
    plt.axis('off')
    plt.show()