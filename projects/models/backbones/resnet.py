import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("../")
from _utils import transfer_params_model


def conv3x3(in_channel, out_channel, stride=1, dilation_rate=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride, padding=dilation_rate, dilation=dilation_rate, bias=False)


def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 1, stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, inter_channel, stride, dilation_rate):
        super(Bottleneck, self).__init__()
        self.conv0 = conv1x1(in_channel, inter_channel)
        self.bn0 = nn.BatchNorm2d(inter_channel)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inter_channel, inter_channel, stride, dilation_rate)
        self.bn1 = nn.BatchNorm2d(inter_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(inter_channel, inter_channel * self.expansion)
        self.bn2 = nn.BatchNorm2d(inter_channel * self.expansion)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if in_channel != inter_channel * 4 or stride != 1:
            self.shortcut = nn.Sequential(
                conv1x1(in_channel, inter_channel * self.expansion, stride),
                nn.BatchNorm2d(inter_channel * self.expansion)
            )
    
    def forward(self, x):
        y = self.relu0(self.bn0(self.conv0(x)))
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.relu2(self.bn2(self.conv2(y)) + self.shortcut(x))
        return y


class ResNet(nn.Module):
    def __init__(self, res_unit, num_units, is_dilate, num_classes=1000, return_dict=False):
        super(ResNet, self).__init__()
        self.dilation_rate = 1
        self.in_channel = 64
        self.return_dict = return_dict
        self.conv0 = nn.Conv2d(3, 64, (7, 7), stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block0 = self._build_block(res_unit, 64, num_units[0], is_dilate[0])
        self.block1 = self._build_block(res_unit, 128, num_units[1], is_dilate[1], stride=2)
        self.block2 = self._build_block(res_unit, 256, num_units[2], is_dilate[2], stride=2)
        self.block3 = self._build_block(res_unit, 512, num_units[3], is_dilate[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512 * res_unit.expansion, num_classes)

    def _build_block(self, res_unit, inter_channel, num_unit, is_dilate=False, stride=1):
        _dilation_rate = self.dilation_rate
        if is_dilate:
            self.dilation_rate *= stride
            stride = 1
        blocks = [res_unit(self.in_channel, inter_channel, stride, _dilation_rate)]
        self.in_channel = inter_channel * res_unit.expansion
        for _ in range(1, num_unit):
            blocks.append(res_unit(self.in_channel, inter_channel, 1, self.dilation_rate))
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.relu0(self.bn0(self.conv0(x)))
        x = self.maxpool(x)
        feature0 = self.block0(x)
        feature1 = self.block1(feature0)
        feature2 = self.block2(feature1)
        feature3 = self.block3(feature2)
        logits = self.avgpool(feature3)
        logits = self.linear(torch.flatten(logits, 1))
        if self.return_dict:
            return_dict_ = {
                "block0": feature0, "block1": feature1,
                "block2": feature2, "block3": feature3,
                "out": logits
            }
            return return_dict_
        else:

            return logis


def _resnet(arch, res_unit, num_units, is_dilate, transfer_params=True, pretrained=False, progress=True, **kwargs):
    model = ResNet(res_unit, num_units, is_dilate, **kwargs)
    if pretrained:
        state_dict = torch.load(args.job_dir)
        model.load_state_dict(state_dict)
    elif transfer_params:
        state_dict = transfer_params_model(arch, progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(**kwargs):
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], [False, False, True, True], **kwargs)


def test():
    model = resnet50(return_dict=True)
    out = model(torch.rand((2, 3, 224, 224)))
    for k, v in out.items():
        print(k, v.shape)
