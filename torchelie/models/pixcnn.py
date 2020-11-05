import torch
import torch.nn as nn
import torch.nn.functional as F
import torchelie.nn as tnn


class FactoredPredictor(nn.Module):
    def __init__(self, hid_ch, out_ch, n_pred):
        """
        Initialize the bias.

        Args:
            self: (todo): write your description
            hid_ch: (int): write your description
            out_ch: (str): write your description
            n_pred: (int): write your description
        """
        super(FactoredPredictor, self).__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hid_ch + i, hid_ch + i),
                nn.ReLU(inplace=True),
                nn.Linear(hid_ch + i, out_ch)
            )
            for i in range(n_pred)
        ])


    def forward(self, x, y):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            y: (todo): write your description
        """
        # NCHW -> NWHC
        x = x.transpose(1, -1)
        y = y.transpose(1, -1)
        # NWKHC
        out = torch.stack([
            self.heads[i](torch.cat([x, y[..., :i]], dim=-1))
            for i in range(len(self.heads))
        ], dim=2)
        # NCKHW
        return out.transpose(1, -1)


    def sample(self, x, temp):
        """
        Sample samples from a sample

        Args:
            self: (todo): write your description
            x: (int): write your description
            temp: (int): write your description
        """
        sampled = torch.empty(x.shape[0], 0, device=x.device)
        for i in range(len(self.heads)):
            logits = self.heads[i](torch.cat([x, self.normalize(sampled)], dim=1)) / temp
            samp = torch.distributions.Categorical(logits=logits, validate_args=True).sample((1,))
            samp = samp.t()
            sampled = torch.cat([sampled, self.cls_to_val(samp.float())], dim=1)
        return sampled


class PixelPredictor(FactoredPredictor):
    def __init__(self, hid_ch, n_ch=3):
        """
        Initialize the chaining.

        Args:
            self: (todo): write your description
            hid_ch: (int): write your description
            n_ch: (int): write your description
        """
        super(PixelPredictor, self).__init__(hid_ch, 256, n_ch)

    def normalize(self, x):
        """
        Normalize x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return x * 2 - 1

    def cls_to_val(self, cls):
        """
        Convert the value to a float.

        Args:
            self: (todo): write your description
            cls: (todo): write your description
        """
        return cls.float() / 255


class ResBlk(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, ks, sz):
        """
        Initialize a batch.

        Args:
            self: (todo): write your description
            in_ch: (int): write your description
            hid_ch: (int): write your description
            out_ch: (str): write your description
            ks: (int): write your description
            sz: (array): write your description
        """
        super(ResBlk, self).__init__()
        self.go = tnn.CondSeq(
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=False),
                tnn.Conv1x1(in_ch, hid_ch),

                nn.BatchNorm2d(hid_ch),
                nn.ReLU(inplace=True),
                tnn.TopLeftConv2d(hid_ch, hid_ch, ks, center=True, bias=sz),

                nn.BatchNorm2d(hid_ch),
                nn.ReLU(inplace=True),
                tnn.Conv1x1(hid_ch, out_ch),
        )

    def condition(self, z):
        """
        Set the condition for the given z.

        Args:
            self: (todo): write your description
            z: (todo): write your description
        """
        self.go.condition(z)

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        out = self.go(x)
        return x + out


# https://arxiv.org/pdf/1701.05517.pdf
class PixCNNBase(nn.Module):
    def __init__(self, in_ch, hid, out_ch, quant_lvls, sz, n_layer=3):
        """
        Initialize the layer.

        Args:
            self: (todo): write your description
            in_ch: (int): write your description
            hid: (int): write your description
            out_ch: (str): write your description
            quant_lvls: (todo): write your description
            sz: (array): write your description
            n_layer: (todo): write your description
        """
        super(PixCNNBase, self).__init__()
        self.sz = sz
        self.lin = tnn.CondSeq(
                tnn.TopLeftConv2d(in_ch, hid, 5, center=False, bias=sz),
                nn.ReLU(inplace=True)
        )


        sz2 = sz[0] // 2, sz[1] // 2
        sz4 = sz[0] // 4, sz[1] // 4
        self.l1 = nn.Sequential(*[ResBlk(hid, hid *2 , hid, 5, sz) for _ in range(n_layer)])
        self.l2 = nn.Sequential(*[ResBlk(hid, hid *2 , hid, 5, sz2) for _ in range(n_layer)])
        self.l3 = nn.Sequential(*[ResBlk(hid, hid *2 , hid, 5, sz4) for _ in range(n_layer)])
        self.l4 = nn.Sequential(*[ResBlk(hid, hid *2 , hid, 5, sz4) for _ in range(n_layer)])
        self.l4 = nn.Sequential(*[ResBlk(hid, hid *2 , hid, 5, sz4) for _ in range(n_layer)])
        self.l5 = nn.Sequential(*[ResBlk(hid * 2, hid *4 , hid*2, 5, sz2) for _ in range(n_layer)])
        self.l6 = nn.Sequential(*[ResBlk(hid*3, hid *6 , hid * 3, 5, sz) for _ in range(n_layer)])

        self.lout = PixelPredictor(hid * 3, out_ch)

    def _body(self, x):
        """
        Return the body.

        Args:
            self: (todo): write your description
            x: (int): write your description
        """
        x = self.lin(x)

        x1 = self.l1(x)
        x2 = self.l2(x1[..., ::2, ::2])
        x3 = self.l3(x2[..., ::2, ::2])

        x4 = self.l4(x3)
        x5 = self.l5(torch.cat([F.interpolate(x4, scale_factor=2), x2], dim=1))
        x6 = self.l6(torch.cat([F.interpolate(x5, scale_factor=2), x1], dim=1))
        return F.relu(x6)


    def forward(self, x):
        """
        Implement forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x6 = self._body(x)
        return self.lout(x6, x)


    def sample_xy(self, x, coord_x, coord_y, temp):
        """
        Sample x y coordinate space.

        Args:
            self: (todo): write your description
            x: (int): write your description
            coord_x: (array): write your description
            coord_y: (todo): write your description
            temp: (todo): write your description
        """
        x6 = self._body(x)
        return self.lout.sample(x6[:, :, coord_y, coord_x], temp)


class PixelCNN(PixCNNBase):
    """
    A PixelCNN model with 6 blocks

    Args:
        hid (int): the number of hidden channels in the blocks
        sz ((int, int)): the size of the images to learn. Must be square
        channels (int): number of channels in the data. 3 for RGB images
    """
    def __init__(self, hid, sz, channels=3, n_layer=3):
        """
        Initialize the layer.

        Args:
            self: (todo): write your description
            hid: (int): write your description
            sz: (array): write your description
            channels: (list): write your description
            n_layer: (todo): write your description
        """
        super(PixelCNN, self).__init__(channels, hid, channels, 256, sz,
                n_layer=n_layer)
        self.channels = channels

    def forward(self, x):
        """A forward pass for training"""
        return super().forward(x)

    def sample(self, temp, N):
        """
        Sample a batch of images

        Args:
            temp (float): the sampling temperature
            N (int): number of images to generate in the batch

        Returns:
            A batch of images
        """
        img = torch.zeros(N, self.channels, *self.sz,
                device=next(self.parameters()).device).uniform_(0, 1)
        return self.sample_(img, temp)

    def partial_sample(self, x, temp):
        """
        Sample a sample of samples

        Args:
            self: (todo): write your description
            x: (array): write your description
            temp: (todo): write your description
        """
        x[:, :, x.shape[2] // 2, :] = 0
        return self.sample_(x, temp, start_coord=(x.shape[2] // 2, 0))

    def sample_cond(self, cond, temp):
        """
        Sample conditional conditional conditional conditional conditional conditional conditional conditional.

        Args:
            self: (todo): write your description
            cond: (todo): write your description
            temp: (todo): write your description
        """
        img = torch.empty(cond.shape[0], self.sz[0], *self.sz,
                device=self.biases[0].device).uniform_(0, 1)
        cond_rsz = F.interpolate(cond, size=img.shape[2:], mode='nearest')
        img = torch.cat([img, cond_rsz], dim=1)
        return self.sample_(img, temp)[:, cond.shape[1]:]


    def sample_(self, img, temp=0, start_coord=(0, 0)):
        """
        Sample image

        Args:
            self: (todo): write your description
            img: (array): write your description
            temp: (int): write your description
            start_coord: (int): write your description
        """
        self.eval()
        with torch.no_grad():
            for l in range(start_coord[0], self.sz[0]):
                for c in range(start_coord[1] if l == start_coord[0] else 0,  self.sz[0]):
                    x = self.sample_xy(img * 2 - 1, c, l, temp)
                    img[:, :, l, c] = x

        return img

