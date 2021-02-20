import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
from tensorflow.keras.applications import resnet

model = resnet.ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
# model.summary(line_length=160)

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./test/tb_vis/tf_resnet50")
# x_train = tf.random.normal((16, 224, 224, 3))
# y_train = tf.random.normal((16, 1))
# model.fit(
#     x_train[0:2],
#     y_train[0:2],
#     epochs=1,
#     callbacks=[tensorboard_callback]
# )
def replace_name(name, original, new):
    for orig, new_ in zip(original, new):
        if orig in name:
            name = name.replace(orig, new_)
    return name

scope = sub_scope = m = new_scope = new_sub_scope = new_m = ""
for l in model.layers:
    name = l.name
    name = replace_name(
        name,
        ["block1", "block2", "block3", "block4", "block5", "block6"],
        ["bottleneck0/", "bottleneck1/", "bottleneck2/", "bottleneck3/", "bottleneck4/", "bottleneck5/"]
    )
    name = replace_name(
        name,
        ["_1_conv", "_1_bn", "_1_relu", "_2_conv", "_2_bn", "_2_relu", "_3_conv", "_3_bn", "_out", "_0_conv", "_0_bn"],
        ["conv0", "bn0", "relu0", "conv1", "bn1", "relu1", "conv2", "bn2", "relu2", "DS/conv", "DS/bn"]
    )
    name = replace_name(
        name,
        ["conv1_", "conv2_", "conv3_", "conv4_", "conv5_"],
        ["", "block0/", "block1/", "block2/", "block3/"]
    )
    # print(name)
    if "/" in name:
        new_scope, new_sub_scope, new_m = name.split("/", 2)
        if scope != new_scope:
            print("\n"+new_scope+"\n\t"+new_sub_scope+"\n\t\t", end="")
            scope = new_scope
            sub_scope = new_sub_scope
        elif scope == new_scope and sub_scope != new_sub_scope:
            print("\n\t"+new_sub_scope+"\n\t\t", end="")
            sub_scope = new_sub_scope
        else:
            print("\n\t\t", end="")

        if isinstance(l, layers.Conv2D):
            stride = l.get_config()["strides"][0]
            dilate = l.get_config()["dilation_rate"][0]
            print(
                new_m+f"\tstrides={stride}, rate={dilate}, ",
                l.input.shape[1:], "--", l.weights[0].shape,
                "-->", l.output.shape[1:], sep="", end=""
            )
        else:
            print(new_m, end="")
    else:
        if isinstance(l, layers.Conv2D):
            stride = l.get_config()["strides"][0]
            dilate = l.get_config()["dilation_rate"][0]
            print(
                name+f", strides={stride}, rate={dilate},",
                l.input.shape[1:], "--", l.weights[0].shape,
                "-->", l.output.shape[1:]
            )
        else:
            print(name)

