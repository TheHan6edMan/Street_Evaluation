import tensorflow as tf
try:
    import tensorflow.python.keras.layers as layers
except:
    import tensorflow.keras.layers as layers
try:
    import tensorflow.python.keras.models as models
except:
    import tensorflow.keras.models as models

import tensorflow.keras as keras
import tensorflow.nn as nn


cfg = {
    'resnet20': []
}


def conv3x3(num_filters, strids=1):
    return layers.Conv2D(
        num_filters, kernel_size=3, strides=strides, padding="same",
        data_format="channels_first", use_bias=False, 
    )


class RCUBlock(keras.Model):
    def __init__(self, num_filters, upsample_size=None, **kwargs):
        super(RCUBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3(num_filters)
        self.conv2 = conv3x3(num_filters)
        self.relu = layers.ReLU()
        self.upsample_size = upsample_size
    
    def call(self, x):
        out = self.conv2(self.relu(self.conv1(self.relu(x)))) + x
        if self.upsample_size is not None:
            self.upsample = layers.UpSampling2D(self.upsample_size, "channels_first", interpolation="bilinear")
            out = self.upsample(out)
        return out


class CRPBlock(keras.Model):
    def __init__(self, num_filters):
        super(CRPBlock, self).__init__()
        self.conv1 = conv3x3(num_filters)
        self.conv2 = conv3x3(num_filters)
        self.conv3 = conv3x3(num_filters)
        self.pool1 = layers.MaxPooling2D(5, 1, "same", "channels_first")
        self.pool2 = layers.MaxPooling2D(5, 1, "same", "channels_first")
        self.pool3 = layers.MaxPooling2D(5, 1, "same", "channels_first")
        self.relu = layers.ReLU()
    
    def call(self, x):
        x = self.relu(x)
        out = self.conv1(self.pool1(x))
        x += out
        out = self.conv1(self.pool2(x))
        x += out
        out = self.conv1(self.pool3(x))
        x += out
        return x


class RefineNetBlock(keras.Model):
    def __init__(self, num_filters, num_path, **kwargs):
        super(RefineNetBlock, self).__init__(**kwargs)
        if num_path == 1:
            self.rcu = RCUBlock(num_filters)
        else:
            for i in range(num_path):

        self.crp = CRPBlock(num_filters)
    
    def call(self, x):
        



class LambdaLayer(keras.Model):
    def __init__(self, lamda):
        super(LambdaLayer, self).__init__()
        self.lamda = lamda
    

    def call(self, x):
        return self.lamda(x)


class ResNetBlock(keras.Model):
    expansion = 1
    def __init__(self, num_filters, strides=1, option="B", **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3(num_filters, strides=strides)
        self.conv2 = conv3x3(num_filters)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        if strides != 1:
            if option == "A":
                self.residual = LambdaLayer(lambda x: tf.pad(
                    x[:, :, ::2, ::2],
                    [[0, 0], [num_filters//4, num_filters//4], [0, 0], [0, 0]],
                    mode="CONSTANT", constant_values=0
                ))
            elif option == "B":
                self.residual = keras.Sequential()
                self.residual.add(
                    conv3x3(num_filters)
                    layers.BatchNormalization()
                )


    def call(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)) + self.residual(x))
        return out


class RefineNet(keras.Model):
    def __init__(self, block, num_filters, num_blocks, **kwargs):
        super(RefineNet, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(64, 7, 2, "same", "channels_first", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.resnet_block1 = self._make_resnet_block(block, num_filters[0], num_blocks[0], stride=1)
        self.resnet_block2 = self._make_resnet_block(block, num_filters[1], num_blocks[1], stride=2)
        self.resnet_block3 = self._make_resnet_block(block, num_filters[2], num_blocks[2], stride=2)


    def call(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        b1 = self.resnet_block1(x)
        b2 = self.resnet_block2(b1)
        b3 = self.resnet_block3(b2)


    

    def _make_resnet_block(self, block, num_filters, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = keras.Sequential()
        for stride in strides:
            blocks.add(block(num_filters, stride))



