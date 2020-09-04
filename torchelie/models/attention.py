import functools

import torchelie.utils as tu
import torch.nn as nn
import torchelie.nn as tnn
from .classifier import Classifier1

Block = functools.partial(tnn.PreactResBlock, bottleneck=True)


class UBlock(nn.Module):
    def __init__(self, ch, inner, with_skip=True):
        super(UBlock, self).__init__()
        self.inner = inner
        if with_skip and inner is not None:
            self.skip = Block(ch, ch)
        else:
            self.skip = None
        self.encode = tnn.CondSeq(nn.MaxPool2d(3, 1, 1),
                                  nn.UpsamplingBilinear2d(scale_factor=0.5),
                                  Block(ch, ch))
        self.decode = tnn.CondSeq(Block(ch, ch),
                                  nn.UpsamplingBilinear2d(scale_factor=2))

    def forward(self, x):
        e = self.encode(x)
        if self.inner is not None:
            e2 = self.inner(e)
        else:
            e2 = e

        if self.skip is not None:
            e2 += self.skip(e)

        return self.decode(e2)


class UBlock1(nn.Module):
    def __init__(self, ch):
        super(UBlock1, self).__init__()
        self.inner = tnn.CondSeq(nn.MaxPool2d(3, 1, 1),
                                 nn.UpsamplingBilinear2d(scale_factor=0.5),
                                 Block(ch, ch),
                                 nn.UpsamplingBilinear2d(scale_factor=2))

    def forward(self, x):
        return self.inner(x)


class AttentionBlock(nn.Module):
    def __init__(self,
                 ch,
                 n_down,
                 n_trunk=2,
                 n_post=1,
                 n_pre=1,
                 n_att_conv=2,
                 with_skips=True):
        super(AttentionBlock, self).__init__()
        self.pre = tnn.CondSeq(*[Block(ch, ch) for _ in range(n_pre)])
        self.post = tnn.CondSeq(*[Block(ch, ch) for _ in range(n_post)])
        self.trunk = tnn.CondSeq(*[Block(ch, ch) for _ in range(n_trunk)])

        soft = UBlock1(ch)
        for _ in range(n_down - 1):
            soft = UBlock(ch, soft, with_skip=with_skips)
        if n_down >= 0:
            conv1 = [soft]
            for i in range(n_att_conv):
                conv1 += [
                    nn.BatchNorm2d(ch),
                    nn.ReLU(True),
                    tu.kaiming(tnn.Conv1x1(ch, ch, bias=(i != n_att_conv - 1)))
                ]
            conv1.append(nn.Sigmoid())

            self.mask = tnn.CondSeq(*conv1)
        else:
            self.mask = None

    def forward(self, x):
        x = self.pre(x)
        t = self.trunk(x)
        if self.mask is not None:
            t = t * (self.mask(x) + 1)
        return self.post(t)


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


def attention56(num_classes, in_ch=3):
    """
    Build a attention56 network

    Args:
        num_classes (int): number of classes
        in_ch (int): number of channels in the images
    """
    return Classifier1(Attention56Bone(in_ch), 2048, num_classes=num_classes)
