import random
import time

import numpy as np

v1 = np.random.randn(3)
m1 = np.random.randn(2, 3)
print(m1[0][0] * v1[0] + m1[0][1] * v1[1] + m1[0][2] * v1[2])

print(m1 @ v1)
