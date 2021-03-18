import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50


class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel, dilation_rates):
        super(ASPP, self).__init__()
        self.atrous_conv0 = self._make_branch(in_channel, out_channel, 0, ksize=1)
        self.atrous_conv1 = self._make_branch(in_channel, out_channel, dilation_rates[0])
        self.atrous_conv2 = self._make_branch(in_channel, out_channel, dilation_rates[1])
        self.atrous_conv3 = self._make_branch(in_channel, out_channel, dilation_rates[2])
        self.atrous_pool0 = self._make_branch(in_channel, out_channel, 0, ksize=1, img_pooling=True)


    def _make_branch(self, in_channel, out_channel, dilation_rate, ksize=3, img_pooling=False):
        branch = [
            nn.Conv2d(in_channel, out_channel, ksize, padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True)
        ]
        if img_pooling:
            branch = [nn.AdaptiveAvgPool2d(output_size=1)] + branch
        return nn.Sequential(*branch)

    def forward(self, x):
        size = x.shape[-2:]
        feat0 = self.atrous_conv0(x)
        feat1 = self.atrous_conv1(x)
        feat2 = self.atrous_conv2(x)
        feat3 = self.atrous_conv3(x)
        feat4 = self.atrous_pool0(x)
        feat4 = F.interpolate(feat4, size=size, mode="bilinear", align_corners=False)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), dim=1)
        return y


class DeepLabV3Classifier(nn.Sequential):
    def __init__(self, in_channel, dilation_rates=[6, 12, 18], inter_channel=256, n_classes=21):
        super(DeepLabV3Classifier, self).__init__(
            ASPP(in_channel, inter_channel, dilation_rates),
            nn.Conv2d(inter_channel * 5, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, n_classes, 1)
        )

    def forward(self, x):
        for module in self:
            x = module(x)
        return x


def deeplabv3_classifier(in_channel, n_classes=21, dilation_rates=[6, 12, 18], **kwargs):
    return DeepLabV3Classifier(in_channel, dilation_rates=dilation_rates, n_classes=n_classes, **kwargs)


def test():
    classifier = deeplabv3_classifier(2048)
    y = classifier(torch.rand((2, 2048, 28, 28)))
    print(y.shape)


if __name__ == "__main__":
    test()