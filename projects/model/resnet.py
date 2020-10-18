import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.nn as nn
from tensorflow.keras import Model



class Block(Model):
    def __init__(self, num_filters, strides=1, option="A", **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(num_filters, 3, strides=strides, padding="same", use_bias=False)
        self.conv2 = layers.Conv2D(num_filters, 3, strides=1, padding="same", use_bias=False)
        self.relu1 = layers.ReLU()
        self.relu2 = layers.ReLU()
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        if strides != 1:
            self.residual = 



