import math
import torch
import torch.nn as nn
import torchelie.nn as tnn
import torchelie.utils as tu


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride=1, mul_factor=6):
        super(MBConv, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ks = ks
        self.stride = stride
        self.factor = mul_factor

        hid = in_ch * mul_factor
        self.branch = tnn.CondSeq(
            tu.xavier(tnn.Conv1x1(in_ch, hid, bias=False)),
            nn.BatchNorm2d(hid),
            tnn.HardSwish(),
            tu.xavier(
                nn.Conv2d(hid,
                          hid,
                          ks,
                          stride=stride,
                          padding=ks // 2,
                          groups=hid,
                          bias=False)),
            nn.BatchNorm2d(hid),
            tnn.HardSwish(),
            tnn.SEBlock(hid, reduction=4),
            tu.xavier(tnn.Conv1x1(hid, out_ch)),
            nn.BatchNorm2d(out_ch)
        )

        self.shortcut = tnn.CondSeq()

        if stride != 1:
            self.shortcut.add_module('pool', nn.AvgPool2d(stride, stride,
                ceil_mode=True))

        if in_ch != out_ch:
            self.shortcut.add_module('conv', tnn.Conv1x1(in_ch, out_ch,
                bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_ch))

    def __repr__(self):
        return "MBConv({}, {}, factor={}, k{}x{}s{}))".format(
                self.in_ch, self.out_ch, self.factor, self.ks, self.ks,
                self.stride)

    def forward(self, x):
        return self.branch(x).add_(self.shortcut(x))


class EfficientNet(tnn.CondSeq):
    def __init__(self, in_ch, num_classes, B=0):
        def ch(ch):
            return int(ch * 1.1 ** B) // 8 * 8

        def l(d):
            return int(math.ceil(d * 1.2 ** B))

        def r():
            return int(224 * 1.15 ** B)

        super(EfficientNet, self).__init__(
            #Stage 1
            #nn.UpsamplingBilinear2d(size=(r(), r())),
            tu.kaiming(tnn.Conv3x3(in_ch, ch(32), stride=2, bias=False)),
            nn.BatchNorm2d(ch(32)),
            tnn.HardSwish(),

            #Stage 2
            MBConv(ch(32), ch(16), 3, mul_factor=1),
            *[MBConv(ch(16), ch(16), 3, mul_factor=1) for _ in range(l(1) - 1)],

            #Stage 3
            MBConv(ch(16), ch(24), 3, stride=2),
            *[MBConv(ch(24), ch(24), 3) for _ in range(l(2) - 1)],

            #Stage 4
            MBConv(ch(24), ch(40), 5, stride=2),
            *[MBConv(ch(40), ch(40), 5) for _ in range(l(2) - 1)],

            #Stage 5
            MBConv(ch(40), ch(80), 3, stride=2),
            *[MBConv(ch(80), ch(80), 3) for _ in range(l(3) - 1)],

            #Stage 6
            MBConv(ch(80), ch(112), 5),
            *[MBConv(ch(112), ch(112), 5) for _ in range(l(3) - 1)],

            #Stage 7
            MBConv(ch(112), ch(192), 5, stride=2),
            *[MBConv(ch(192), ch(192), 5) for _ in range(l(4) - 1)],

            #Stage 8
            MBConv(ch(192), ch(320), 3),
            *[MBConv(ch(320), ch(320), 3) for _ in range(l(1) - 1)],

            tu.kaiming(tnn.Conv1x1(ch(320), ch(1280), bias=False)),
            nn.BatchNorm2d(ch(1280)),
            tnn.HardSwish(),
            nn.AdaptiveAvgPool2d(1),
            tnn.Reshape(-1),
            tu.xavier(nn.Linear(ch(1280), num_classes))
        )
