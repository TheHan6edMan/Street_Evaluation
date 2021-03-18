import torch
import torch.nn as nn
from collections import OrderedDict
from torch.hub import load_state_dict_from_url as load_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'deeplabv3_resnet50': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}
map_rules_resnet18 = {
    "layer1": "block0", "layer2": "block1", "layer3": "block2", "layer4": "block3",
    "conv1": "conv0", "conv2": "conv1", "conv3": "conv2", "downsample": "shortcut",
    "bn1": "bn0", "bn2": "bn1", "bn3": "bn2", "fc": "linear",
}
map_rules_deeplabv3_resnet50 = {**map_rules_resnet18,
    "aux_classifier": None, "classifier.1": None, "classifier.2": None,
    "classifier.4": None, "convs.0": "atrous_conv0", "convs.1": "atrous_conv1",
    "convs.2": "atrous_conv2", "convs.3": "atrous_conv3", "convs.4": "atrous_pool0", 
    "classifier.0.project.0": "classifier.1", "classifier.0.project.1": "classifier.2",
    
}
map_rules_resnet101 = map_rules_resnet50 = map_rules_resnet34 = map_rules_resnet18
map_rules_deeplabv3_resnet101 = map_rules_deeplabv3_resnet50


def transfer_params(arch, progress=True):
    from collections import OrderedDict

    def map_name(name, map_rules):
        for old_part, new_part in map_rules.items():
            if old_part in name:
                if new_part is None:
                    return None
                else:
                    name = name.replace(old_part, new_part)
        return name

    state_dict_torch = load_url(model_urls[arch], progress=progress)
    map_rules = eval("map_rules_"+arch)
    state_dict = OrderedDict()
    for name, module in state_dict_torch.items():
        name = map_name(name, map_rules)
        if name:
            state_dict[name] = module
    return state_dict


class InterLayer(nn.ModuleDict):
    def __init__(self, model, top_layer, return_layers=[]):
        if not set(return_layers + [top_layer]).issubset([name for name, _ in model.named_children()]):
            raise ValueError("top_layer are not present in model")
        self.top_layer = top_layer
        self.return_layers = return_layers
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name == top_layer:
                break
        super(InterLayer, self).__init__(layers)

    def forward(self, x):
        feat_dict = dict()
        for name, module in self.items():
            print(name, "\n", module, flush=True)
            print("-"*128)
            x = module(x)
            if name in self.return_layers:
                feat_dict[name] = x
        return feat_dict if len(self.return_layers) > 0 else x
