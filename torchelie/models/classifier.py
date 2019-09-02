import torch
import torch.nn as nn

import torchelie.nn as tnn
from torchelie.utils import kaiming, xavier


class Classifier(nn.Module):
    """
    A classification head added on top of a feature extraction model.

    Args:
        feat_extractor (nn.Module): a feature extraction model
        feature_size (int): the number of features in the last layer of the
            feature extractor
        num_classes (int): the number of output classes
    """
    def __init__(self, feat_extractor, feature_size, num_classes):
        super(Classifier, self).__init__()
        self.bone = feat_extractor

        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            tnn.Reshape(feature_size),
            kaiming(nn.Linear(feature_size, feature_size)),
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


