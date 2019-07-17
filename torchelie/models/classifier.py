import torch
import torch.nn as nn

import torchelie.nn as tnn
from torchelie.utils import kaiming, xavier


class Classifier(nn.Module):
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
        return self.head(self.bone(*xs))


class ProjectionDiscr(nn.Module):
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
        feats = self.head(self.bone(x, y))
        y_emb = self.emb(y)
        return self.discr(feats) + torch.mm(y_emb, feats.t())


