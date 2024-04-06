import numpy as np
import keras

bob = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
jeff = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

joe = np.matmul(bob, jeff)

george = np.array([1, 2, 3])

print(np.matmul(george, jeff))

clifford = [np.array([5, 10, 15, 22]), np.array([15, 29, 44, 59]), np.array([29, 50, 70, 100])]

loss = np.sum(np.subtract(clifford, joe) ** 2) / len(bob)
print(loss)

loss2 = 0
for i in range(len(bob)):
    jill = np.matmul(bob[i], jeff)
    loss2 += np.sum(np.subtract(clifford[i], jill) ** 2)
loss2 /= len(bob)
print(loss2)

print(np.sum(george, axis=0))


(train_values, train_labels), (test_values, test_labels) = keras.datasets.mnist.load_data()


train_values = np.append(train_values, test_values, axis=0)
print(len(train_values))
