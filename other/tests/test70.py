import numpy as np


def matmul_chn(a, b):
    c1 = a * b[0]
    c2 = a * b[1]
    c1 = np.sum(c1, axis=0)
    c2 = np.sum(c2, axis=0)
    return np.append(c1, c2, axis=0)


arr1 = np.ones((5, 3, 2))
arr2 = np.ones((2, 5))

print('1:')
print(arr1)
print(arr1.shape)
print(arr1.T.shape)
print()
print('2:')
print(arr2)
print(arr2.shape)
print()

print('parts:')
print(f"1-{arr1[-1]}")
print(f"2-{arr2[-2]}")
print()

print('3:')
# last axis arr1, 2nd last axis arr2
print(np.dot(arr1, arr2))
print(np.dot(arr1, arr2).shape)

print('4:')
print(matmul_chn(arr1, arr2))
