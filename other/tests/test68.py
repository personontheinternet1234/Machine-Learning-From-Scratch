# jacobian test 2

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

c = a @ b

ja = np.kron(np.eye(a.shape[0]), b.T)
jb = np.kron(np.eye(b.shape[0]), a.T)

print(ja.shape)
print(ja)
print(jb.shape)
print(jb)
