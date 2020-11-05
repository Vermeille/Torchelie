import torch
import torch.nn as nn

import torchelie.nn as tnn
from torchelie.utils import kaiming, xavier


class Classifier2(nn.Module):
    """
    A classification head added on top of a feature extraction model.

    Args:
        feat_extractor (nn.Module): a feature extraction model
        feature_size (int): the number of features in the last layer of the
            feature extractor
        num_classes (int): the number of output classes
    """
    def __init__(self, feat_extractor, feature_size, num_classes):
        """
        Initialize the graph.

        Args:
            self: (todo): write your description
            feat_extractor: (todo): write your description
            feature_size: (int): write your description
            num_classes: (int): write your description
        """
        super(Classifier2, self).__init__()
        self.bone = feat_extractor

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            tnn.Reshape(feature_size),
            kaiming(nn.Linear(feature_size, feature_size)),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            xavier(nn.Linear(feature_size, num_classes)),
        )

    def forward(self, *xs):
        """
        Forward pass

        Args:
            *xs: arguments for `feat_extractor`
        """
        out = self.head(self.bone(*xs))
        return out


class Classifier1(nn.Module):
    """
    A one layer classification head added on top of a feature extraction model.

    Args:
        feat_extractor (nn.Module): a feature extraction model
        feature_size (int): the number of features in the last layer of the
            feature extractor
        num_classes (int): the number of output classes
    """
    def __init__(self, feat_extractor, feature_size, num_classes, dropout=0.5):
        """
        Initialize the classifier.

        Args:
            self: (todo): write your description
            feat_extractor: (todo): write your description
            feature_size: (int): write your description
            num_classes: (int): write your description
            dropout: (str): write your description
        """
        super(Classifier1, self).__init__()
        self.bone = feat_extractor

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            tnn.Reshape(feature_size),
            nn.Dropout(dropout),
            kaiming(nn.Linear(feature_size, num_classes)),
        )

    def forward(self, *xs):
        """
        Forward pass

        Args:
            *xs: arguments for `feat_extractor`
        """
        out = self.head(self.bone(*xs))
        return out


class ConcatPoolClassifier1(nn.Module):
    """
    A one layer classification head added on top of a feature extraction model.
    it includes ConcatPool.

    Args:
        feat_extractor (nn.Module): a feature extraction model
        feature_size (int): the number of features in the last layer of the
            feature extractor
        num_classes (int): the number of output classes
    """
    def __init__(self, feat_extractor, feature_size, num_classes, dropout=0.5):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            feat_extractor: (todo): write your description
            feature_size: (int): write your description
            num_classes: (int): write your description
            dropout: (str): write your description
        """
        super(ConcatPoolClassifier1, self).__init__()
        self.bone = feat_extractor

        self.head = nn.Sequential(
            tnn.AdaptiveConcatPool2d(1),
            tnn.Reshape(feature_size * 2),
            nn.Dropout(dropout),
            kaiming(nn.Linear(feature_size * 2, num_classes)),
        )

    def forward(self, *xs):
        """
        Forward pass

        Args:
            *xs: arguments for `feat_extractor`
        """
        out = self.head(self.bone(*xs))
        return out


class ProjectionDiscr(nn.Module):
    """
    A classification head for conditional GANs discriminators using a
    projection discriminator from https://arxiv.org/abs/1802.05637

    Args:
        feat_extractor (nn.Module): a feature extraction model
        feature_size (int): the number of features in the last layer of the
            feature extractor
        num_classes (int): the number of output classes
    """
    def __init__(self, feat_extractor, feature_size, num_classes):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            feat_extractor: (todo): write your description
            feature_size: (int): write your description
            num_classes: (int): write your description
        """
        super(ProjectionDiscr, self).__init__()
        self.bone = feat_extractor
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            tnn.Reshape(feature_size),
        )
        self.emb = nn.Embedding(num_classes, feature_size)
        self.discr = nn.Linear(feature_size, 1)

    def forward(self, x, y):
        """
        Forward pass

        Args:
            x: argument for `feat_extractor`
            y: class label
        """
        feats = self.head(self.bone(x, y))
        y_emb = self.emb(y)
        return self.discr(feats) + torch.mm(y_emb, feats.t())


