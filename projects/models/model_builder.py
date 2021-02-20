import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as seg_models

import backbones
import classifiers


class SegModel(nn.Module):
    def __init__(self, backbone, model_arch, in_channel, num_classes, **kwargs):
        super(SegModel, self).__init__()
        self.backbone = backbones.__dict__[backbone](return_dict=True)
        self.classifier = classifiers.__dict__[model_arch+"_classifier"](in_channel=in_channel, num_classes=num_classes, **kwargs)
    
    def forward(self, x):
        img_size = x.shape[-2:]
        x = self.backbone(x)
        x = self.classifier(x)
        x = F.interpolate(x, img_size, mode="bilinear", align_corners=False)
        return x


def _load_model(backbone, model_arch, in_channel, num_classes, pretrained=False, progress=True, **kwargs):
    model = SegModel(backbone, model_arch, in_channel, num_classes, **kwargs)
    if pretrained:
        pass
    return model


def deeplabv3_resnet50(pretrained=False, progress=True, num_classes=21, **kwargs):
    return _load_model("resnet50", "deeplabv3", 2048, num_classes, pretrained, progress, **kwargs)


def test():
    model = deeplabv3_resnet50()
    y = model(torch.rand((2, 3, 224, 224)))
    model_torch = seg_models.deeplabv3_resnet50(aux_loss=True)
    print(model.classifier)
    print("="*128)
    print(model_torch.classifier)


test()