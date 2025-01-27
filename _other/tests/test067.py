# jacobian test 1

import numpy as np

arr1 = np.random.randn(2, 3)
arr2 = np.random.randn(3, 4)

res1 = arr1 @ arr2

jacob1 = np.kron(np.eye(arr1.shape[0]), arr2.T)
jacob2 = np.kron(np.eye(arr2.shape[0]), arr1.T)

print(jacob1.shape)
print(jacob1)
print(jacob2.shape)
print(jacob2)
