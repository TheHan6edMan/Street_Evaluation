import tensorflow as tf
import tensorflow.python.keras.layers as layers


x = tf.constant(range(1, 51), shape=[1, 5, 5, 2], dtype=tf.float32)

conv = layers.Conv2D(16, 3, padding="same")
y = conv(x)

class Test():
    def __init__(self):
        self.item1 = 2

    def _test(self, i):
        setattr(self, "item", i)

tst = Test()
tst._test(2)
print(tst.item)
