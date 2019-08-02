import torch
import torch.nn as nn
import torch.nn.functional as F
import torchelie.nn as tnn


class ResBlk(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, ks, norm=nn.BatchNorm2d):
        super(ResBlk, self).__init__()
        self.go = tnn.CondSeq(
                tnn.MConvNormReLU(in_ch, hid_ch, ks, norm),
                tnn.MConvNormReLU(hid_ch, out_ch, ks, norm),
        )
        self.shortcut = None
        if in_ch != out_ch:
            self.shortcut = tnn.CondSeq(
                    tnn.Conv1x1(in_ch, out_ch),
                    nn.BatchNorm(out_ch)
            )

    def condition(self, z):
        self.go.condition(z)

    def forward(self, x):
        out = self.go(x)
        if self.shortcut:
            x = self.shortcut(x)
        return x + out


# https://arxiv.org/pdf/1701.05517.pdf
class PixCNNBase(nn.Module):
    def __init__(self, in_ch, hid, out_ch, sz):
        super(PixCNNBase, self).__init__()
        self.sz = sz
        self.lin = tnn.MConvBNrelu(in_ch, hid, 5, center=False)
        self.bias = nn.Parameter(torch.zeros(hid, *sz))

        self.l1 = ResBlk(hid, hid *2 , hid, 3)
        self.l2 = ResBlk(hid, hid *2 , hid, 3)
        self.l3 = ResBlk(hid, hid *2 , hid, 3)
        self.l4 = ResBlk(hid, hid *2 , hid, 3)
        self.l5 = ResBlk(hid, hid *2 , hid, 3)
        self.l6 = ResBlk(hid, hid *2 , hid, 3)


        self.lout = nn.Sequential(
            tnn.MaskedConv2d(hid, out_ch, 3, center=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                #m.weight.data *= 2
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.lin(x) + self.bias

        x1 = self.l1(x) #1
        x2 = self.l2(F.interpolate(x1, scale_factor=0.5)) #2
        x3 = self.l3(F.interpolate(x2, scale_factor=0.5)) #3

        x4 = self.l4(x3) #3
        x5 = self.l5(F.interpolate(x4, scale_factor=2) + x2)
        x6 = self.l6(F.interpolate(x5, scale_factor=2) + x1)
        return self.lout(x6)


class SimplePixCNN(nn.Module):
    def __init__(self, in_ch, hid, out_ch, sz):
        super(SimplePixCNN, self).__init__()
        self.sz = sz
        self.lin = tnn.CondSeq(
                tnn.MaskedConv2d(in_ch, hid, 5, center=False),
                nn.ReLU(inplace=True),
        )
        self.bias = nn.Parameter(torch.zeros(hid, *sz))
        self.body = tnn.CondSeq(
                #nn.Dropout(),
                tnn.MaskedConv2d(hid, hid, 5, center=True),
                nn.BatchNorm2d(hid),
                nn.ReLU(inplace=True),
                #nn.Dropout(),
                tnn.MaskedConv2d(hid, hid, 5, center=True),
                nn.BatchNorm2d(hid),
                nn.ReLU(inplace=True),
                tnn.MaskedConv2d(hid, hid, 5, center=True),
                #nn.Dropout(),
                nn.BatchNorm2d(hid),
                nn.ReLU(inplace=True),
            )
        self.lout = nn.Sequential(
            tnn.MaskedConv2d(hid, out_ch, 5, center=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.weight.data *= 2
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.lin(x) + self.bias

        x = self.body(x)
        return self.lout(x)


class PixelCNN(PixCNNBase):
    def __init__(self, hid, sz, channels=3):
        super(PixelCNN, self).__init__(channels, hid, 256 * channels, sz)
        self.channels = channels

    def forward(self, x):
        log_probs = super().forward(x)
        B, C, H, W = log_probs.shape
        return log_probs.view(B, 256, self.channels, H, W)

    def sample(self, temp, N):
        img = torch.zeros(N, self.channels, *self.sz, device=self.bias.device).uniform_(0, 1)
        return self.sample_(img, temp)


    def sample_cond(self, cond, temp):
        img = torch.empty(cond.shape[0], self.bias.shape[0], *self.sz,
                device=self.bias.device).uniform_(0, 1)
        cond_rsz = F.interpolate(cond, size=img.shape[2:], mode='nearest')
        img = torch.cat([img, cond_rsz], dim=1)
        return self.sample_(img, temp)[:, cond.shape[1]:]


    def sample_(self, img, temp=0):
        self.eval()
        with torch.no_grad():
            for l in range(self.sz[0]):
                for c in range(self.sz[0]):
                    log_prob = self(img * 2 - 1)[:, :, :, l, c] / temp
                    for i in range(img.shape[0]):
                        logits = (log_prob[i].transpose(0, 1))
                        x = torch.distributions.Categorical(logits=logits, validate_args=True).sample((1,))
                        img[i, :, l, c] = x.float() / 255

        print(img[0])
        return img

