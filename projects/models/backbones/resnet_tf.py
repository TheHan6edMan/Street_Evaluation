import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def conv3x3(num_filters, stride=1, dilation_rate=1, **kwargs):
    return layers.Conv2D(num_filters, 3, stride, "same", use_bias=False, dilation_rate=dilation_rate, **kwargs)

def conv1x1(num_filters, stride=1, dilation_rate=1, **kwargs):
    return layers.Conv2D(num_filters, 1, stride, "same", use_bias=False, dilation_rate=dilation_rate, **kwargs)

def print_conv(conv):
    return f"{conv.name}(stride={conv.strides[0]}, dilation={conv.dilation_rate[0]}, padding={conv.padding})"


class Bottleneck(keras.Model):
    expansion = 4

    def __init__(self, inter_channel, stride=1, dilation_rate=1, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.stride = stride

        self.conv0 = conv1x1(inter_channel, name="conv0")
        self.bn0 = layers.BatchNormalization(epsilon=1.001e-5, name="bn0")
        self.relu0 = layers.Activation("relu", name="relu0")
        self.conv1 = conv3x3(inter_channel, stride, dilation_rate, name="conv1")
        self.bn1 = layers.BatchNormalization(epsilon=1.001e-5, name="bn1")
        self.relu1 = layers.Activation("relu", name="relu1")
        self.conv2 = conv1x1(inter_channel * 4, name="conv2")
        self.bn2 = layers.BatchNormalization(epsilon=1.001e-5, name="bn2")

        self.add = layers.Add(name="add")
        self.relu2 = layers.Activation("relu", name="relu2")        
    
    def call(self, x):
        shortcut = x
        x = self.relu0(self.bn0(self.conv0(x)))
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if shortcut.shape[-1] != x.shape[-1]:
            if not hasattr(self, "downsample"):
                self.downsample = keras.Sequential(
                    [conv1x1(x.shape[-1], self.stride, name="conv_ds"),
                    layers.BatchNormalization(epsilon=1.001e-5, name="bn_ds")],
                    name="downsample"
                )
            shortcut = self.downsample(shortcut)
        y = self.relu2(self.add([x, shortcut]))
        return y


class ResNet(keras.Model):
    def __init__(self, res_unit, nunit_per_block, dilate_config=None, num_classes=1000, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.dilation_rate = 1
        if dilate_config is None:
            dilate_config = [False, False, False, False]
        if len(dilate_config) != 4:
            raise "`dilate_config` should be None or a 4-element tuple, got %d" % len(dilate_config)
        if dilate_config[0] is not False:
            raise "the 1st elemnt of `dilate_config` has to be False"
        self.conv = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False, name="conv")
        self.bn = layers.BatchNormalization(epsilon=1.001e-5, name="bn")
        self.relu = layers.Activation("relu", name="relu")
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding="same", name="maxpool")
        self.block0 = self._make_block(res_unit, 64, nunit_per_block[0], dilate=dilate_config[0], name="block0")
        self.block1 = self._make_block(res_unit, 128, nunit_per_block[1], stride=2, dilate=dilate_config[1], name="block1")
        self.block2 = self._make_block(res_unit, 256, nunit_per_block[2], stride=2, dilate=dilate_config[2], name="block2")
        self.block3 = self._make_block(res_unit, 512, nunit_per_block[3], stride=2, dilate=dilate_config[3], name="block3")
        self.avgpool = layers.GlobalAvgPool2D(name="global_avgpool")
        self.dense = layers.Dense(num_classes, name="dense")
    
    def _make_block(self, res_unit, inter_channel, num_unit, stride=1, dilate=False, name=None):
        dilation_rate_ = self.dilation_rate
        if dilate:
            self.dilation_rate *= stride
            stride = 1
        blocks = keras.Sequential(res_unit(inter_channel, stride, dilation_rate_, name="bottleneck0"), name=name)
        for i in range(1, num_unit):
            blocks.add(res_unit(inter_channel, 1, self.dilation_rate, name="bottleneck"+str(i)))
        return blocks
    
    def call(self, x, **kwargs):
        x = self.relu(self.bn(self.conv(x)))
        x = self.maxpool(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        y = self.dense(self.avgpool(x))
        return y

    # @tf.function
    # def train_step(self, data):
    #     if len(data) == 3:
    #         x, y, sample_weight = data
    #     else:
    #         (x, y), sample_weight = data, None
    #     print(x.shape, y.shape)
    #     with tf.GradientTape() as tape:
    #         y_pred = self(x)
    #         print("y_pred:", y_pred.shape, end="\t")
    #         loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #         print("loss:", loss.numpy())
    #     grads = tape.gradient(loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    #     self.compiled_metrics.update_state(y, y_pred)
    #     return {met.name: met.result() for met in self.metrics}


def resnet50(input_shape=(224, 224, 3), pretrained=True, use_tf_model=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], [False, False, True, True], name="resnet50", **kwargs)
    model.build((None,)+input_shape)
    if pretrained:
        pass
    if use_tf_model:
        tf_model = keras.applications.ResNet50(include_top=True, input_shape=input_shape)
        named_modules = {m.name: m for m in tf_model.submodules if len(m.weights) > 0}
        mapping_part = {
            "conv0": "1_conv", "conv1": "2_conv", "conv2": "3_conv", "conv_ds": "0_conv",
            "bn0": "1_bn", "bn1": "2_bn", "bn2": "3_bn", "bn_ds": "0_bn", "downsample/": "",
            "block0/": "conv2_", "block1/": "conv3_", "block2/": "conv4_", "block3/": "conv5_",
            "bottleneck0/": "block1_", "bottleneck1/": "block2_", "bottleneck2/": "block3_",
            "bottleneck3/": "block4_", "bottleneck4/": "block5_", "bottleneck5/": "block6_",
        }
        mapping_name = {"conv": "conv1_conv", "bn": "conv1_bn", "dense": "predictions"}
        model = transfer_params(model, named_modules, mapping_part, mapping_name, include_bias=False)
    return model

def test():
    model = resnet50()


def is_transfer(model, named_modules, mapping_part, mapping_name):

    def map_name(name):
        for orig, new in mapping_part.items():
            if orig in name:
                name = name.replace(orig, new)
        for orig, new in mapping_name.items():
            if name == orig:
                name = new
        return name
    
    def _is_transfer(module, name=""):
        if isinstance(module, layers.Conv2D):
            name_ = map_name(name)
            print(tf.reduce_all(module.kernel == named_modules[name_].kernel))
        elif isinstance(module, layers.BatchNormalization):
            name_ = map_name(name)
            print(tf.reduce_all(module.moving_mean == named_modules[name_].moving_mean))
            print(tf.reduce_all(module.moving_variance == named_modules[name_].moving_variance))
            if module.gamma is not None:
                print(tf.reduce_all(module.gamma == named_modules[name_].gamma))
                print(tf.reduce_all(module.beta == named_modules[name_].beta))
        elif isinstance(module, layers.Dense):
            name_ = map_name(name)
            print(tf.reduce_all(module.kernel == named_modules[name_].kernel))
        if hasattr(module, "layers"):
            for submodule in module.layers:
                _is_transfer(submodule, name=name+"/"+submodule.name)

    for module in model.layers:
        _is_transfer(module, module.name)


def transfer_params(model, named_modules, mapping_part, mapping_name, include_bias=False):

    def map_name(name):
        for orig, new in mapping_part.items():
            if orig in name:
                name = name.replace(orig, new)
        for orig, new in mapping_name.items():
            if name == orig:
                name = new
        return name
    
    def _transfer_params(module, name=""):
        if isinstance(module, layers.Conv2D):
            name_ = map_name(name)
            assert module.kernel.shape == named_modules[name_].kernel.shape
            module.kernel = named_modules[name_].kernel
            if include_bias:
                module.bias = named_modules[name_].bias
        elif isinstance(module, layers.BatchNormalization):
            name_ = map_name(name)
            module.moving_mean = named_modules[name_].moving_mean
            module.moving_variance = named_modules[name_].moving_variance
            if module.gamma is not None:
                module.gamma = named_modules[name_].gamma
                module.beta = named_modules[name_].beta
        elif isinstance(module, layers.Dense):
            name_ = map_name(name)
            assert module.kernel.shape == named_modules[name_].kernel.shape
            module.kernel = named_modules[name_].kernel
            module.bias = named_modules[name_].bias
        if hasattr(module, "layers"):
            for submodule in module.layers:
                _transfer_params(submodule, name=name+"/"+submodule.name)

    for module in model.layers:
        _transfer_params(module, module.name)
    return model


test()

