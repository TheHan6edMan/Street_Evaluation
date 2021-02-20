import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel, dilation_rates):
        super(ASPP, self).__init__()
        self.atrous_conv0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU()
        )
        self.atrous_conv1 = self._make_atrous_conv_branch(in_channel, out_channel, dilation_rates[0])
        self.atrous_conv2 = self._make_atrous_conv_branch(in_channel, out_channel, dilation_rates[1])
        self.atrous_conv3 = self._make_atrous_conv_branch(in_channel, out_channel, dilation_rates[2])
        self.atrous_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def _make_atrous_conv_branch(self, in_channel, out_channel, dilation_rate):
        branch = [
            nn.Conv2d(in_channel, out_channel, 3, padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU()
        ]
        return nn.Sequential(*branch)

    def forward(self, x):
        size = x.shape[-2:]
        feature0 = self.atrous_conv0(x)
        feature1 = self.atrous_conv1(x)
        feature2 = self.atrous_conv2(x)
        feature3 = self.atrous_conv3(x)
        feature4 = self.atrous_pool(x)
        feature4 = F.interpolate(feature4, size=size, mode="bilinear", align_corners=False)
        y = torch.cat((feature0, feature1, feature2, feature3, feature4), dim=1)
        return y


class DeepLabV3Classifier(nn.Module):
    def __init__(self, in_channel, dilation_rates=[6, 12, 18], inter_channel=256, num_classes=21):
        super(DeepLabV3Classifier, self).__init__()
        self.classifier = nn.Sequential(
            ASPP(in_channel, inter_channel, dilation_rates),
            nn.Conv2d(inter_channel * 5, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        return self.classifier(x)


def deeplabv3_classifier(in_channel, num_classes=21, dilation_rates=[6, 12, 18], **kwargs):
    return DeepLabV3Classifier(in_channel, dilation_rates=dilation_rates, num_classes=num_classes, **kwargs)


def test():
    classifier = deeplabv3_classifier(2048)
    y = classifier(torch.rand((2, 2048, 28, 28)))
    print(y.shape)
