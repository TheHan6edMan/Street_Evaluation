# The Cityscapes Dataset
本仓库包含了用于检查、准备和评估 Cityscapes 数据集的脚本；该数据集包含了从 50 个不同城市的街景中记录的一组的立体视频序列，具有 5000 帧的高质量像素级注释，以及包含一组含有 20000 个弱注释的帧；

更多详情及下载请浏览[官网](www.cityscapes-dataset.net)


## Dataset Structure

Cityscapes数据集的文件夹结构为`"{root}/{type}{video}/{split}/{city}/{city}_{seq:0>6}_{frame:0>6}_{type}{ext}"`
其中每个部分的含义为
- `root`: Cityscapes 表示数据集根目录的文件夹，很多脚本会检查指向该文件夹的环境变量`CITYSCAPES_DATASET`是否存在，并将其作为默认选项
- `type`: 数据的类型/形式，例如`gtFine`代表细粒度标注的 ground truth，`leftImg8bit`代表 left 8-bit images；
- `split`: 数据集的分割，例如`train/val/test/train_extra/demoVideo`；请注意，并非所有类型的数据都存在于所有的分割中；
- `city`：采集这部分数据集的城市
- `seq`：6 位数字的序列号
- `frame`: 6 位数的帧号；需要注意的是，一些城市记录了很少但很长的视频序列，而在一些城市则记录了很多短视频序列，这些只有第 19 帧用来注释；
- `ext`：文件的扩展名和后缀，例如`_polygons.json`代表了 ground truth 文件；



**`type`的可能取值：**

 - `gtFine`：细粒度的标注，其中包含了 2975 个训练数据、500 个验证数据和 1525 测试数据；标注是使用`json`文件编码的，这些文件含有单独的 polygon 的；此外，数据集所提供的`png`图像的像素值是标签的编码；更多详细内容请参阅`helpers/labels.py`和`preparation`中的脚本；
 - `gtCoarse`：可用于训练和验证的粗粒度标注，以及另一组 1998 年采集的的训练图像 (`train_extra`)；这些注释可以用于模型训练，或与`gtFine`一起使用，或单独用于弱监督学习；
 - `gtBbox3d`：车辆的三维边框标注，更多细节参见 [Cityscapes 3D (Gählert et al., CVPRW '20)](https://arxiv.org/abs/2006.07864)；
 - `gtBboxCityPersons`：行人边界框注释，适用于所有训练和验证图像，边框的四个值是`(x, y, w, h)`，其中`(x, y)`是边框的左上角坐标，`(w, h)`是边框的宽度和高度；更多细节参见`helpers/labels_cityPersons.py`和 [CityPersons (Zhang et al., CVPR '17)](https://bitbucket.org/shanshanzhang/citypersons)；
 - `leftImg8bit`：LDR 格式的 8 比特 left images，这些是标准的注释图像；
 - `leftImg8bit_blurred`：对人脸和车牌照进行模糊处理的 DR 格式的 8 比特 left images；建议在原始图像上计算结果，但使用模糊的图像进行可视化；此处感谢 [Mapillary](https://www.mapillary.com/) 的模糊处理；
 - `leftImg16bit` the left images in 16-bit HDR format. 这些图像每像素为 16 位色深且包含更多信息，尤其是场景中非常暗或亮的部分；需要警惕的是，图像存以 16 位 png 格式储存并非标准格式，进而并非所有库都支持；
 - `rightImg8bit`：8 比特 LDR 格式的右侧立体视图
 - `rightImg16bit`：16 比特 LDR 格式的右侧立体视图
 - `timestamp`：以 $n$ s 表示的记录时间；每个序列的第一帧总是有一个 0 的时间戳 (timestamp)；
 - `disparity`：：预先计算的视差深度图；可以用像素值 p > 0 来计算视差：`d = (float(p) - 1) / 256`；`p = 0`是无效的度量;需要警惕的是，图像存以 16 位 png 格式储存并非标准格式，进而并非所有库都支持；
 - `camera`：内外相机校准；更多细节参见 [csCalibration.pdf](docs/csCalibration.pdf)
 - `vehicle`：车辆里程表 (vehicle odometry)，GPS 坐标和室外温度；更多细节参见 [csCalibration.pdf](docs/csCalibration.pdf)；



**`split`的可能取值：**

- `train`：常用于训练，包含 2975 张细粒度和粗粒度标注的图像；
- `val`：用于超参数的验证，包含 500 张细粒度和粗粒度标注的图像；也可以用作训练；
- `test`：用于在官方的评估服务器上进行测试，其标注不公开，但为了方便，其包含了 ego-vehicle 和整改边界标注；
- `train_extra`：需要时可用于训练，包含 19998 张带有粗粒度标注的图像；
- `demoVideo`：用于定性评估的视频序列，这些视频没有注释；




## Scripts

### Installation

- 利用`pip`安装`cityscapesscripts`
    ```
    python -m pip install cityscapesscripts
    ```

- 图形查看工具和标注工具是基于 Qt5 开发的，可以通过以下命令行安装：
    ```
    python -m pip install cityscapesscripts[gui]
    ```

### Usage

模块`cityscapesscripts`开放以下接口

- `csDownload`: 通过命令行下载 Cityscapes 包；
- `csViewer`: 查看图像并 overlay 注释；
- `csLabelTool`: 用于标记的工具；
- `csEvalPixelLevelSemanticLabeling`: 在验证集或测试集上评估像素级语义分割的结果
- `csEvalInstanceLevelSemanticLabeling`: 在验证集或测试集上评估实例分割的结果
- `csEvalPanopticSemanticLabeling`: 在验证集或测试集上评估全景分割 (panoptic segmentation) 的结果
- `csEvalObjectDetection3d`: 在验证集或测试集上评估 3D 目标检测的结果
- `csCreateTrainIdLabelImgs`: 将 polygonal 格式的标注转换为带有标签 ID 的`png`图像，其中像素编码为“train ID”，该编码可以在`labels.py`中定义 Convert annotations in polygonal format to png images with label IDs, where pixels encode "train IDs" that you can define in `labels.py`.
- `csCreateTrainIdInstanceImgs`: 将 polygonal 格式的标注转换为带有实例 ID 的`png`图像，其中像素编码为由“train ID”组成的实例 ID；
- `csCreatePanopticImgs`: 将标准`png`格式的标注转换为[COCO 全景分割格式](http://cocodataset.org/#format-data).
- `csPlot3dDetectionResults`: 对以`.json`格式保存的 3D 目标检测的测评估结果进行可视化；


### Package Content

程序包的结构如下所述：
- `annotation`: 用于给数据集标注的标注工具
- `download`: Cityscapes 包的下载工具
- `evaluation`: 验证模型
- `helpers`: 其他脚本所包含的帮助文件
- `preparation`: 将 ground truth 转换为适合个人模型的格式
- `viewer`: 查看图片和标注

需要说明的是，所有文件的顶部都有备注文档；核心文件如下：
 - `helpers/labels.py`: 定义所有语义类的 ID，并提供各类属性之间的映射的文件；
 - `helpers/labels_cityPersons.py`: file defining the IDs of all CityPersons pedestrian classes and providing mapping between various class properties.
 - `setup.py`: 运行`CYTHONIZE_EVAL= python setup.py build_ext --inplace`来启用 cython 插件加速评估，效果仅在 Ubuntu 上测试过


## Evaluation

Once you want to test your method on the test set, please run your approach on the provided test images and submit your results:
[Submission Page](www.cityscapes-dataset.net/submit)

The result format is described at the top of our evaluation scripts:
- [Pixel Level Semantic Labeling](cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py)
- [Instance Level Semantic Labeling](cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py)
- [Panoptic Semantic Labeling](cityscapesscripts/evaluation/evalPanopticSemanticLabeling.py)
- [3D Object Detection](cityscapesscripts/evaluation/evalObjectDetection3d.py)

Note that our evaluation scripts are included in the scripts folder and can be used to test your approach on the validation set. For further details regarding the submission process, please consult our website.
