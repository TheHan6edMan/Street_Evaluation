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
import os




def conv3x3(nfilters, strides=1):
    return layers.Conv2D(
        nfilters, kernel_size=3, strides=strides, padding="same",
        data_format="channels_first", use_bias=False, 
    )


class RCUBlock(keras.Model):
    def __init__(self, nfilters, npath=2, nsublayer=2, **kwargs):
        super(RCUBlock, self).__init__(**kwargs)
        for i in range(npath):
            for j in range(nsublayer):
                setattr(self, f"conv{i+1}_{j+1}", conv3x3(nfilters))
        self.relu = layers.ReLU()
        self.npath = npath
        self.nsublayer = nsublayer
    
    def call(self, x):
        if self.npath == 1:
            out = x
            for j in range(self.nsublayer):
                out = getattr(self, f"conv{i+1}_{j+1}")(self.relu(out))
            out += x
        else:
            assert isinstance(x, (list, tuple))
            maxh = tf.reduce_max([xi.shape[-2] for xi in x])
            maxw = tf.reduce_max([xi.shape[-1] for xi in x])
            out = 0
            for i in range(self.npath):
                outi = x[i]
                for j in range(self.nsublayer):
                    outi = getattr(self, f"conv{i+1}_{j+1}")(self.relu(outi))
                outi += x[i]
                assert maxh % out.shape[-2] == 0
                assert maxw % out.shape[-1] == 0
                upsample_size = (maxh / xi.shape[-2], maxw / xi.shape[-1])
                if upsample_size != (1, 1):
                    self.upsample = layers.UpSampling2D(upsample_size, "channels_first", interpolation="bilinear")
                    outi = self.upsample(outi)
                out += outi
        return out


class CRPBlock(keras.Model):
    def __init__(self, nfilters, nsublayer, **kwargs):
        super(CRPBlock, self).__init__(**kwargs)
        for i in range(nsublayer):
            setattr(self, f"conv{i+1}", conv3x3(nfilters))
            setattr(self, f"pool{i+1}", layers.MaxPooling2D(5, 1, "same", "channels_first"))
        self.relu = layers.ReLU()
        self.nsublayer = nsublayer
    
    def call(self, x):
        x = self.relu(x)
        for i in range(self.nsublayer):
            out = getattr(self, f"pool{i+1}")(x)
            out = getattr(self, f"conv{i+1}")(out)
            x += out
        return x


class RefineNetBlock(keras.Model):
    def __init__(self, nfilters, npath, **kwargs):
        super(RefineNetBlock, self).__init__(**kwargs)
        self.cru1 = RCUBlock(nfilters, npath)
        self.cru2 = RCUBlock(nfilters, npath=1, nsublayer=1)
        self.crp = CRPBlock(nfilters)
    
    def call(self, x):
        out = self.cru2(self.crp(self.cru1(x)))
        return out
        

class LambdaLayer(keras.Model):
    def __init__(self, lamda):
        super(LambdaLayer, self).__init__()
        self.lamda = lamda

    def call(self, x):
        return self.lamda(x)


class ResNetBlock(keras.Model):
    expansion = 1
    def __init__(self, nfilters, strides=1, option="B", **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3(nfilters, strides=strides)
        self.conv2 = conv3x3(nfilters)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        if strides != 1:
            if option == "A":
                self.residual = LambdaLayer(lambda x: tf.pad(
                    x[:, :, ::2, ::2],
                    [[0, 0], [nfilters//4, nfilters//4], [0, 0], [0, 0]],
                    mode="CONSTANT", constant_values=0
                ))
            elif option == "B":
                self.residual = keras.Sequential()
                self.residual.add(conv3x3(nfilters))
                self.residual.add(layers.BatchNormalization())


    def call(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)) + self.residual(x))
        return out


class RefineNet(keras.Model):
    def __init__(self, block, nfilters, nblocks, **kwargs):
        super(RefineNet, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(64, 7, 2, "same", "channels_first", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.resnet_block1 = self._make_resnet_block(nfilters[0], nblocks[0], stride=1)
        self.resnet_block2 = self._make_resnet_block(nfilters[1], nblocks[1], stride=2)
        self.resnet_block3 = self._make_resnet_block(nfilters[2], nblocks[2], stride=2)

    def _make_refinenet_block(self, block, nfilters, npath, nsublayer):
        blocks = keras.Sequential()
        blocks.add(RCUBlock(nfilters, npath))
        blocks.add(CRPBlock(nfilters))
        blocks.add(RCUBlock(nfilters, npath=1, nsublayer=nsublayer))
        return blocks

    def _make_resnet_block(self, nfilters, nblocks, stride=1):
        strides = [stride] + [1] * (nblocks - 1)
        blocks = keras.Sequential()
        for stride in strides:
            blocks.add(ResNetBlock(nfilters, stride))
        return blocks
    
    def call(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        b1 = self.resnet_block1(x)
        b2 = self.resnet_block2(b1)
        b3 = self.resnet_block3(b2)

def test():
    ckpt_dir = "./third_party_lib/checkpoints/RefineNet_step_100000.ckpt.index"
    print(os.path.abspath(ckpt_dir))
    model = RefineNet(ResNetBlock, [64, 128, 256], [3, 3, 3])
    model.load_weights(ckpt_dir)

test()

