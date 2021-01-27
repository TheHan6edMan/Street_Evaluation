import tensorflow as tf
import tensorflow.python.keras.layers as layers


x = tf.constant(range(1, 51), shape=[1, 5, 5, 2], dtype=tf.float32)

out = 0
for i in range(3):
    outi = tf.constant(range(1, 7), shape=[2, 3])
    out += outi
print(out)
# out = tf.reduce_sum(out, axis=0)
# print(out)
