import numpy as np

arr1 = np.random.randn(3, 4, 4)
arr2 = np.random.randn(2, 4)

arr3 = np.dot(arr2, arr1)
arr4 = arr2 @ arr1
print(arr1.shape)
print(arr1)
print()
print(arr2.shape)
print(arr2)
print()
print(arr3.shape)
print(arr3)
print()
print(arr4.shape)
print(arr4)
