import tensorflow as tf
import tensorflow.nn as nn
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow.keras import Model



class Layer(Model):
    def __init__(self, num_filter):
        super(Layer, self).__init__()
        self.conv = layers.Conv2D(filters=num_filter, kernel_size=(3, 3), padding="same")
        self.relu = layers.ReLU()
        self.bn = layers.BatchNormalization()
    
    def call(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class Block(Model):
    def __init__(self, num_filter, num_layer):
        super(Block, self).__init__()
        self.block = models.Sequential()
        for i in range(num_layer):
            self.block.add(Layer(num_filter))
        self.block.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"))
    
    def call(self, x):
        return nn.dropout(self.block(x), 0.2)


class Vgg(Model):
    def __init__(self, config):
        super(Vgg, self).__init__()
        num_filter = 64
        self.block = models.Sequential()
        for num_layer in config:
            self.block.add(Block(num_filter, num_layer))
            num_filter *= 2
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation="relu")
        self.fc2 = layers.Dense(512, activation="relu")
        self.fc3 = layers.Dense(10, activation="softmax")
    
    def call(self, x):
        x = self.block(x)
        x = self.flatten(x)
        x = nn.dropout(self.fc1(x), 0.2)
        x = nn.dropout(self.fc2(x), 0.2)
        return self.fc3(x)
        

def vgg16():
    return Vgg([2] * 2 + [3] * 3)

def test():
    model = vgg16()
    print(model)

test()
