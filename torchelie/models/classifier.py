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

    def forward(self, x):
        return self.head(self.bone(x))
