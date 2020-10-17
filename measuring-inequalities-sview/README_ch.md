此代码用于论文：[Measuring social, environmental and health inequalities using deep learning and street imagery](https://www.nature.com/articles/s41598-019-42036-w)

# 数据准备

## 1. 标注

代码文件：`/data-prep/get_labels.py`

**概述**

用于标注的原始数据(每个较低的超级输出地区 - LSOA)需从下述链接对应的相关网站上下载以下是对应链接，从而代码能够读取这些下载的文件并计算出用于训练标记的十分位数。

* [UK Census 2011](http://infusecp.mimas.ac.uk/): 论文中给出了关于变量的细节性描述，对用于复制分析普查变量内容的代码可以在每个不同的输出变量的脚本中找到。

* [English indices of deprivation 2015](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2015)
* [Household income estimates](https://data.london.gov.uk/dataset/household-income-estimates-small-areas): Available for London only
* [ONS Postcode Directory August 2017](https://ons.maps.arcgis.com/home/item.html?id=151e4a246b91c34178a55aab047413f29b): Links  seem to be unstable, a search for ONS Postcode Directory ArcGIS gives you the up to date link for download


## 2. 提取图像特征 

代码文件：`/data-prep/vgg_features_extract.py`

##### 概述

对于每一个街道层面的图像，此代码利用[ VGG16 网络预训练的权重](https://github.com/machrisaa/tensorflow-vgg)来提取出一个 4096 维的向量

### 3. 创建训练所需的 HDF5 数据 `/data-prep/make_hdf5.py`

从 VGG16 提取的特征和 pickle 文件中创建 hdf5 文件，以用于训练标注

##### 读取

- 利用 VGG16 从图片中提取的特征
- 带有标注的元数据和输入图片的id

##### 输出

- 带有特征的 HDF5 文件
- 带有相应标注值的标注 

## 基于深度学习的分配到十分位数的有序分类

`/classification/ordinal_classification_sview_tboard.py`

将每一个地区的相关结果分配给一个十分位数是一个有序分类任务，因此我们使用以下网络进行分类，其中卷积层部分我们使用了 VGG16 的预训练的权重，进而这一步我们直接训练全链接层的权重


![](classification/fig_nework_github.png)


## 集成(Aggregation)

`/aggregation/get_lsoa_level_predictions.py`

我们分析所需的地面真实数据仅在我们关注的城市的较低的超级输出区域 (LSOA) 能够得到。为测试网络性能，我们需要在 LSOA 层级预测而不是图像层级进行预测，进而我们将训练好的网络最后一层输出的结果在地区层级上进行取平均值，并将该平均值输入 sigmoid 函数，进而将可视为十分位数中的一个类别，进而能够与 LSOA 的实际十分位数进行比较。

## Tensorflow 的预训练模型

Will be made available soon for trained networks using London images
