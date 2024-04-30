import numpy as np
import tensorflow as tf

arraymulti = tf.constant([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

arraymulti2 = tf.constant([
    [1, 2, 2],
    [3, 3, 3],
    [4, 4, 4]
])

# print(tf.math.argmax(arraymulti))

print(tf.cumsum(arraymulti, axis=0)[-1])

print(tf.raw_ops.Concat(0, (arraymulti, arraymulti2)))