import time

import tensorflow as tf

import matplotlib.pyplot as plt
from tqdm import tqdm

lr = 0.01
ep = 5
ins = 2
hls = 3
ots = 2

X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
Y = tf.constant([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=tf.float32)

W0 = tf.Variable(tf.random.uniform((ins, hls)))
B0 = tf.Variable(tf.zeros((1, hls)))
W1 = tf.Variable(tf.random.uniform((hls, ots)))
B1 = tf.Variable(tf.zeros((1, ots)))

t1 = time.time()
Ll = []
for i in tqdm(range(ep), ncols=150):
    # f
    with tf.GradientTape() as tape:
        A1 = tf.nn.leaky_relu(tf.matmul(X, W0) + B0)
        A2 = tf.nn.leaky_relu(tf.matmul(A1, W1) + B1)
        loss = tf.losses.mean_squared_error(Y, A2)

    # b
    gradients = tape.gradient(loss, [W0, B0, W1, B1])
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    optimizer.apply_gradients(zip(gradients, [W0, B0, W1, B1]))

    # e
    Ll.append(loss.numpy())

plt.plot(range(ep), Ll, color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss v.s. Epoch")
plt.show()

while True:
    print("")
    inputs = []
    for i in range(ins):
        inputs.append(float(input(f"{str(i)}: ")))

    # f
    A0_s = np.array(inputs)
    A1_s = l_relu(np.matmul(A0_s, W0) + B0)
    A2_s = l_relu(np.matmul(A1_s, W1) + B1)

    # r
    print("")
    print(A2_s)
