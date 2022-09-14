from collections import OrderedDict

import torch.nn as nn
from .vgg import vgg19
from torchelie.nn.utils import edit_model
from typing import List
from torchelie.nn.withsavedactivations import WithSavedActivations

class PerceptualNet(WithSavedActivations):
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
    def __init__(self,
                 layers: List[str],
                 use_avg_pool: bool = True,
                 remove_unused_layers: bool = True) -> None:
        # yapf: disable
        layer_names = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'maxpool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'maxpool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
                    'conv3_4', 'relu3_4', 'maxpool3',  # noqa: E131
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
                    'conv4_4', 'relu4_4', 'maxpool4',  # noqa: E131
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
                    'conv5_4', 'relu5_4',  # 'maxpool5'
        ]
        # yapf: enable

        m = vgg19(1, pretrained='perceptual/imagenet').features
        flat_vgg = [
            layer for layer in m.modules()
            if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d))
        ]
        m = nn.Sequential(
            OrderedDict([(n, mod) for n, mod in zip(layer_names, flat_vgg)]))
        for nm, mod in m.named_modules():
            if 'relu' in nm:
                setattr(m, nm, nn.ReLU(False))
            elif 'pool' in nm and use_avg_pool:
                setattr(m, nm, nn.AvgPool2d(2, 2))

        if remove_unused_layers:
            m = m[:max([layer_names.index(layer) for layer in layers]) + 1]

        super().__init__(m, names=layers)
