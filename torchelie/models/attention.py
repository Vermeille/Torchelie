import functools

import torchelie.utils as tu
import torch.nn as nn
import torchelie.nn as tnn
from .classifier import Classifier1

Block = functools.partial(tnn.PreactResBlock, bottleneck=True)


class UBlock(nn.Module):
    def __init__(self, ch, inner):
        super(UBlock, self).__init__()
        self.inner = inner
        if inner is not None:
            self.skip = Block(ch, ch)
        self.encode = tnn.CondSeq(nn.MaxPool2d(3, 2, 1), Block(ch, ch))
        self.decode = tnn.CondSeq(Block(ch, ch),
                                  nn.UpsamplingBilinear2d(scale_factor=2))

    def forward(self, x):
        e = self.encode(x)
        if self.inner is not None:
            e2 = self.inner(e) + self.skip(e)
        else:
            e2 = e
        return self.decode(e2)


class AttentionBlock(nn.Module):
    def __init__(self, ch, n_down):
        super(AttentionBlock, self).__init__()
        self.pre = Block(ch, ch)
        self.post = Block(ch, ch)
        self.trunk = tnn.CondSeq(Block(ch, ch), Block(ch, ch))

        soft = None
        for _ in range(n_down):
            soft = UBlock(ch, soft)
        self.mask = tnn.CondSeq(soft, nn.BatchNorm2d(ch), nn.ReLU(True),
                                tu.kaiming(tnn.Conv1x1(ch, ch)),
                                nn.BatchNorm2d(ch), nn.ReLU(True),
                                tu.kaiming(tnn.Conv1x1(ch, ch)),
                                tnn.HardSigmoid())

    def forward(self, x):
        x = self.pre(x)
        t = self.trunk(x)
        m = self.mask(x)
        m.add_(1)
        return self.post(t * m)


class Attention56Bone(nn.Module):
    """
    Attention56 bone

    Args:
        in_ch (int): number of channels in the images
    """
    def __init__(self, in_ch=3):
        super(Attention56Bone, self).__init__()
        self.head = tnn.CondSeq(tu.kaiming(tnn.Conv2d(in_ch, 64, 7, stride=2)),
                                nn.ReLU(True), nn.MaxPool2d(3, 2, 1))
        self.pre1 = Block(64, 256)
        self.attn1 = AttentionBlock(256, 3)
        self.pre2 = Block(256, 512, stride=2)
        self.attn2 = AttentionBlock(512, 2)
        self.pre3 = Block(512, 1024, stride=2)
        self.attn3 = AttentionBlock(1024, 1)
        self.pre4 = tnn.CondSeq(
            Block(1024, 2048, stride=2),
            Block(2048, 2048),
            Block(2048, 2048),
        )

    def forward(self, x):
        x = self.head(x)
        x = self.pre1(x)
        x = self.attn1(x)
        x = self.pre2(x)
        x = self.attn2(x)
        x = self.pre3(x)
        x = self.attn3(x)
        x = self.pre4(x)
        return x


#64:2 M 256 A:3 512::2 A:2 1024:2 A:1 2048:2 2048 2048
def attention56(num_classes, in_ch=3):
    """
    Build a attention56 network

    Args:
        num_classes (int): number of classes
        in_ch (int): number of channels in the images
    """
    return Classifier1(Attention56Bone(in_ch), 2048, num_classes=num_classes)
