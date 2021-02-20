import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url as load_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}

map_rules_resnet18 = {
    "layer1": "block0", "layer2": "block1", "layer3": "block2", "layer4": "block3",
    "conv1": "conv0", "conv2": "conv1", "conv3": "conv2", "downsample": "shortcut",
    "bn1": "bn0", "bn2": "bn1", "bn3": "bn2", "fc": "linear",
}
map_rules_resnet101 = map_rules_resnet50 = map_rules_resnet34 = map_rules_resnet18


def transfer_params_model(arch, progress=True):
    from collections import OrderedDict

    def map_name(name, map_rules):
        for orig, new in map_rules.items():
            if orig in name:
                name = name.replace(orig, new)
        return name

    state_dict = load_url(model_urls[arch], progress=progress)
    map_rules = eval("map_rules_"+arch)
    state_dict_ = OrderedDict()
    for k, v in state_dict.items():
        state_dict_[map_name(k, map_rules)] = v
    return state_dict_
