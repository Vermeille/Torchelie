from collections import OrderedDict

from torchelie.utils import edit_model
import torch.nn as nn
import torchvision.models as M
from torchelie.nn import WithSavedActivations
from typing import List


def PerceptualNet(layers: List[str],
                  use_avg_pool: bool = True,
                  remove_unused_layers: bool = True) -> nn.Module:
    """
    Make a VGG16 with appropriately named layers that records intermediate
    activations.

    Args:
        layers (list of str): the names of the layers for which to save the
            activations.
        use_avg_pool (bool): Whether to replace max pooling with averange
            pooling (default: True)
        remove_unused_layers (bool): whether to remove layers past the last one
            used (default: True)
    """
    # yapf: disable
    layer_names = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'maxpool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'maxpool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
                'conv3_4', 'relu3_4', 'maxpool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
                'conv4_4', 'relu4_4', 'maxpool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
                'conv5_4', 'relu5_4',# 'maxpool5'
    ]
    # yapf: enable

    m = M.vgg19(pretrained=True).eval().features
    m = nn.Sequential(
        OrderedDict([(l_name, l) for l_name, l in zip(layer_names, m)]))
    for nm, mod in m.named_modules():
        if 'relu' in nm:
            setattr(m, nm, nn.ReLU(False))
        elif 'pool' in nm and use_avg_pool:
            setattr(m, nm, nn.AvgPool2d(2, 2))

    if remove_unused_layers:
        m = m[:max([layer_names.index(l) for l in layers]) + 1]

    m = WithSavedActivations(m, names=layers)
    return m


def PaddedPerceptualNet(layers: List[str],
                        use_avg_pool: bool = True) -> nn.Module:
    """
    Make a VGG16 with appropriately named layers that records intermediate
    activations.

    Args:
        layers (list of str): the names of the layers for which to save the
            activations.
    """
    # yapf: disable
    layer_names = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'maxpool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'maxpool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
                'conv3_4', 'relu3_4', 'maxpool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
                'conv4_4', 'relu4_4', 'maxpool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
                'conv5_4', 'relu5_4',# 'maxpool5'
    ]
    # yapf: enable

    m = M.vgg19(pretrained=True).eval().features

    ls = []
    for l_name, l in zip(layer_names, m):
        if isinstance(l, nn.Conv2d):
            ls.append((l_name.replace('conv', 'pad'), nn.ReflectionPad2d(1)))
        ls.append((l_name, l))
    m = nn.Sequential(OrderedDict(ls))

    for nm, mod in m.named_modules():
        if 'relu' in nm:
            setattr(m, nm, nn.ReLU(True))
        elif 'pool' in nm and use_avg_pool:
            setattr(m, nm, nn.AvgPool2d(2, 2))
        elif 'conv' in nm:
            mod.padding = (0, 0)
    m = WithSavedActivations(m, names=layers)
    return m
