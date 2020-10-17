# Tensorflow VGG16 and VGG19

这是基于 [tensorflow-vgg16](https://github.com/ry/tensorflow-vgg16) 和 [Caffe to Tensorflow](https://github.com/ethereon/caffe-tensorflow) 的对 VGG 16 and VGG 19 的实现，在 [here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) 和 [here](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77)可以找到原始的 Caffe 实现

我们对<a href="https://github.com/ry/tensorflow-vgg16">tensorflow-vgg16</a>的实现进行了调整，包括利用 Numpy 不是默认的 TensorFlow 模型加载数据，进而能够加快加载速度并减少整体的内存占用；这个对神经网络的实现支持对网络的进一步调整，例如去掉全连接层或增加 batch 大小

>若要使用 VGG 网络，需要下载 [VGG16 NPY](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) or [VGG19 NPY](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) 的 npy 文件（需要翻墙）

## 使用
利用以下代码创建 VGG 对象
```python
vgg = vgg19.Vgg19()
vgg.build(images)
```
or
```Python
vgg = vgg16.Vgg16()
vgg.build(images)
```
`images`是一个形状为`[None, 224, 224, 3]`的张量

>Trick: 该张量可以是一个 placeholder, variable 甚至是一个 constant.

可以利用 VGG 对象访问所有 VGG 网络层，例如 `vgg.conv1_1`, `vgg.conv1_2`, `vgg.pool5`, `vgg.prob`, ...

`test_vgg16.py` 和 `test_vgg19.py` 包含使用示例

## Extra
This library has been used in my another Tensorflow image style synethesis project: [stylenet](https://github.com/machrisaa/stylenet)


## Update 1: Trainable VGG:

添加了 VGG19 的一个可训练模型`vgg19_trainable`，其支持利用已存在的变量进行训练或从头训练(但不包括训练器)

A very simple testing is added `test_vgg19_trainable`, switch has demo about how to train, switch off train mode for verification, and how to save.

A seperated file is added (instead of changing existing one) because I want to keep the simplicity of the original VGG networks.


## Update 2: Tensorflow v1.0.0:
All the source code has been upgraded to [v1.0.0](https://github.com/tensorflow/tensorflow/blob/v1.0.0-rc1/RELEASE.md).

The conversion is done by my another project [tf0to1](https://github.com/machrisaa/tf0to1)

