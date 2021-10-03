from collections import OrderedDict
import torchelie.nn as tnn
import torch.nn as nn

from .registry import register
from .classifier import ClassificationHead

__all__ = ['AlexNet', 'alexnet', 'alexnet_bn', 'ZFNet', 'zfnet', 'zfnet_bn']


class AlexNet(tnn.CondSeq):

    def __init__(self, num_classes):
        super().__init__()
        self.features = tnn.CondSeq(
            OrderedDict([
                # 224 -> 112 -> 56
                ('conv1', tnn.ConvBlock(3, 64, kernel_size=11, stride=4)),
                # 56 -> 28
                ('pool1', nn.MaxPool2d(3, 2, 1)),
                ('conv2', tnn.ConvBlock(64, 192, kernel_size=5)),
                # 28 -> 14
                ('pool2', nn.MaxPool2d(3, 2, 1)),
                ('conv3', tnn.ConvBlock(192, 384, 3)),
                ('conv4', tnn.ConvBlock(384, 256, 3)),
                ('conv5', tnn.ConvBlock(256, 256, 3)),
                # 28 -> 14
                ('pool3', nn.MaxPool2d(3, 2, 1)),
            ]))
        self.classifier = ClassificationHead(256, num_classes)
        self.classifier = self.classifier.to_two_layers(4096).set_pool_size(7)

    def remove_batchnorm(self):
        for m in self.features:
            if isinstance(m, tnn.ConvBlock):
                m.remove_batchnorm()
        return self


@register
def alexnet(num_classes):
    return AlexNet(num_classes).remove_batchnorm()


@register
def alexnet_bn(num_classes):
    return AlexNet(num_classes)


class ZFNet(tnn.CondSeq):

    def __init__(self, num_classes):
        super().__init__()
        self.features = tnn.CondSeq(
            OrderedDict([
                # 224 -> 112
                ('conv1', tnn.ConvBlock(3, 96, kernel_size=7, stride=2)),
                # 112 -> 56
                ('pool1', nn.MaxPool2d(3, 2, 1)),
                # 56 -> 28
                ('conv2', tnn.ConvBlock(96, 256, kernel_size=5, stride=2)),
                # 28 -> 14
                ('pool2', nn.MaxPool2d(3, 2, 1)),
                ('conv3', tnn.ConvBlock(256, 384, 3)),
                ('conv4', tnn.ConvBlock(384, 256, 3)),
                ('conv5', tnn.ConvBlock(256, 256, 3)),
                # 28 -> 14
                ('pool3', nn.MaxPool2d(3, 2, 1)),
            ]))
        self.classifier = ClassificationHead(256, num_classes)
        self.classifier = self.classifier.to_two_layers(4096).set_pool_size(7)

    def remove_batchnorm(self):
        for m in self.features:
            if isinstance(m, tnn.ConvBlock):
                m.remove_batchnorm()
        return self


@register
def zfnet(num_classes):
    return ZFNet(num_classes).remove_batchnorm()


@register
def zfnet_bn(num_classes):
    return ZFNet(num_classes)
