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
    
    def forward(self, inputs):
        in_size = inputs.shape[-2:]
        feat_dict = self.backbone(inputs)
        x = feat_dict["block3"]
        x = self.classifier(x)
        out = F.interpolate(x, in_size, mode="bilinear", align_corners=False)
        return out


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
    # model_torch = seg_models.deeplabv3_resnet50(aux_loss=True)
    # print(model.classifier)
    # print("="*128)
    # print(model_torch.classifier)

if __name__ == "__main__":
    test()