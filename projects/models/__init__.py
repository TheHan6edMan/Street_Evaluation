import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as seg_models
import os

from . import backbones
from . import classifiers
from ._utils import transfer_params

# import backbones
# import classifiers
# from _utils import transfer_params


class SegModel(nn.Module):
    def __init__(self, backbone, arch, in_channel, n_classes, return_dict=False, **kwargs):
        super(SegModel, self).__init__()
        self.return_dict = return_dict
        self.backbone = backbones.__dict__[backbone](include_top=False, return_dict=True)
        self.classifier = classifiers.__dict__[arch+"_classifier"](in_channel=in_channel, n_classes=n_classes, **kwargs)

    def forward(self, inputs):
        print(inputs)
        in_size = inputs.shape[-2:]
        feat = self.backbone(inputs)
        print("backbone_out")
        backbone_out = feat["block3"] if isinstance(feat, dict) else feat
        classifier_out = self.classifier(backbone_out)
        out = F.interpolate(classifier_out, in_size, mode="bilinear", align_corners=False)
        print("classifier_out")
        if self.return_dict:
            feat = feat if isinstance(feat, dict) else {"block3": feat}
            feat.update({"b4_upsampling": classifier_out, "out": out})
            return feat
        else:
            return out


def _load_model(backbone, arch, in_channel_clf, n_classes, state_dict_dir=None, return_dict=False, **kwargs):
    model = SegModel(backbone, arch, in_channel_clf, n_classes, return_dict=return_dict, **kwargs)
    if state_dict_dir is not None:
        state_dict_dir_ = os.path.join(state_dict_dir, arch+"_"+backbone+".pth")
        if os.path.isfile(state_dict_dir_):
            state_dict = torch.load(state_dict_dir_)
            model.load_state_dict(state_dict)
        else:
            os.makedirs(state_dict_dir, exist_ok=True)
            state_dict = transfer_params(arch+"_"+backbone, progress=True)
            model.load_state_dict(state_dict, strict=False)
            torch.save(model.state_dict(), state_dict_dir_)
    return model


def deeplabv3_resnet50(state_dict_dir=None, n_classes=19, return_dict=False, **kwargs):
    return _load_model("resnet50", "deeplabv3", 2048, n_classes, state_dict_dir, return_dict, **kwargs)


def test():
    model = deeplabv3_resnet50(state_dict_dir="../../experiments/baseline", n_classes=19, return_dict=True)
    out = model(torch.rand((2, 3, 224, 224)))
    for n, t in out.items():
        print(n, t.shape)

if __name__ == "__main__":
    test()