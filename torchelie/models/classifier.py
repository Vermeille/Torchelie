from collections import OrderedDict
import torch
import torch.nn as nn

from torchelie.utils import experimental
import torchelie.nn as tnn
from torchelie.utils import kaiming


class ClassificationHead(tnn.CondSeq):
    """
    A one layer classification head, turning activations / features into class
    log probabilities.

    It initially contains an avgpool-flatten-linear architecture.

    Args:
        in_channels (int): the number of features in the last layer of the
            feature extractor
        num_classes (int): the number of output classes
    """
    linear1: nn.Linear

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(ClassificationHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.to_resnet_style()

    def to_resnet_style(self) -> 'ClassificationHead':
        """
        Set the classifier architecture to avgpool-flatten-linear.
        """
        self._modules = OrderedDict([
            ('pool', nn.AdaptiveAvgPool2d(1)),
            ('reshape', tnn.Reshape(self.in_channels)),
            ('linear1', kaiming(nn.Linear(self.in_channels,
                                          self.num_classes))),
        ])

        return self

    def to_two_layers(self, hidden_channels: int) -> 'ClassificationHead':
        """
        Set the classifier architecture to avgpool-flatten-linear1-relu-linear2.
        """
        self._modules = OrderedDict([
            ('pool', nn.AdaptiveAvgPool2d(1)),
            ('reshape', tnn.Reshape(self.in_channels)),
            ('linear1', kaiming(nn.Linear(self.in_channels, hidden_channels))),
            ('relu1', nn.ReLU(True)),
            ('linear2', kaiming(nn.Linear(hidden_channels, self.num_classes))),
        ])
        return self

    def to_vgg_style(self, hidden_channels: int) -> 'ClassificationHead':
        """
        Set the classifier architecture to
        avgpool-flatten-linear1-relu-dropout-linear2-relu-dropout-linear3, like
        initially done with VGG.
        """
        self._modules = OrderedDict([
            ('pool', nn.AdaptiveAvgPool2d(7)),
            ('reshape', tnn.Reshape(7 * 7 * self.in_channels)),
            ('linear1',
             kaiming(nn.Linear(7 * 7 * self.in_channels, hidden_channels))),
            ('relu1', nn.ReLU(True)),
            ('dropout1', nn.Dropout(0.5)),
            ('linear2', kaiming(nn.Linear(hidden_channels, hidden_channels))),
            ('relu2', nn.ReLU(True)),
            ('dropout2', nn.Dropout(0.5)),
            ('linear3', kaiming(nn.Linear(hidden_channels, self.num_classes))),
        ])

        return self

    def to_convolutional(self) -> 'ClassificationHead':
        """
        Remove pooling and flattening operations, convert linears to conv1x1
        """
        del self.reshape
        del self.pool

        def _do(m):
            if isinstance(m, nn.Linear):
                return tnn.Conv1x1(m.in_features, m.out_features)
            return m
        tnn.utils.edit_model(self, _do)
        return self

    def leaky(self) -> 'ClassificationHead':
        """
        Make relus leaky
        """
        tnn.utils.make_leaky(self)
        return self

    def set_num_classes(self, classes: int) -> 'ClassificationHead':
        """
        change the number of output classes
        """
        self.num_classes = classes
        old = self[-1]
        self[-1] = kaiming(nn.Linear(old.in_features, self.num_classes))
        return self

    def remove_pool(self, spatial_size: int) -> 'ClassificationHead':
        """
        remove the pooling operation
        """
        self.set_pool_size(spatial_size)
        del self.pool
        return self

    def set_pool_size(self, size: int) -> 'ClassificationHead':
        """
        Average pool to spatial size :code:`size` rather than 1. Recreate the
        first Linear to accomodate the change.
        """
        self.pool = nn.AdaptiveAvgPool2d(size)
        self.reshape = tnn.Reshape(size * size * self.in_channels)
        self.linear1 = kaiming(
            nn.Linear(size * size * self.in_channels,
                      self.linear1.out_features))
        return self

    @experimental
    def rm_dropout(self) -> 'ClassificationHead':
        """
        Remove the dropout layers if any.
        """
        if hasattr(self, 'dropout1'):
            del self.dropout1
        if hasattr(self, 'dropout2'):
            del self.dropout2
        return self


@experimental
class ProjectionDiscr(nn.Module):
    """
    A classification head for conditional GANs discriminators using a
    `projection discriminator <https://arxiv.org/abs/1802.05637>`_ .

    Args:
        feat_extractor (nn.Module): a feature extraction model
        feature_size (int): the number of features in the last layer of the
            feature extractor
        num_classes (int): the number of output classes
    """
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            tnn.Reshape(in_channels),
        )
        self.emb = nn.Embedding(num_classes, in_channels)
        self.discr = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: argument for `feat_extractor`
            y: class label
        """
        feats = self.head(x)
        y_emb = self.emb(y)
        return self.discr(feats) + torch.mm(y_emb, feats.t())
