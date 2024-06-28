import time
import numpy as np
arr = np.random.randn(10, 10)

l1 = []
l2 = []
l3 = []
l4 = []
one = time.time()
for i in range(100000):
    l1.append(arr * arr)
two = time.time()
for i in range(100000):
    l2.insert(0, arr * arr)
three = time.time()
for i in range(100000):
    j = arr * arr
    l3.append(j)
four = time.time()
for i in range(100000):
    l4.append(arr * arr)
five = time.time()

print(two - one)
print(three - two)
print(four - three)
print(five - four)
