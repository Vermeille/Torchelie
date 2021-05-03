import torch
import torch.nn as nn
import torch.nn.functional as F
import torchelie.nn as tnn
from torchelie.utils import experimental
from typing import Tuple


class FactoredPredictor(nn.Module):
    heads: nn.ModuleList

    def __init__(self, hid_ch: int, out_ch: int, n_pred: int) -> None:
        super(FactoredPredictor, self).__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_ch + i, hid_ch + i),
                          nn.ReLU(inplace=True), nn.Linear(hid_ch + i, out_ch))
            for i in range(n_pred)
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

    def sample(self, x: torch.Tensor, temp: float) -> torch.Tensor:
        sampled = torch.empty(x.shape[0], 0, device=x.device)
        for i in range(len(self.heads)):
            logits = self.heads[i](torch.cat([x, self.normalize(sampled)],
                                             dim=1)) / temp
            samp = torch.distributions.Categorical(logits=logits,
                                                   validate_args=True).sample(
                                                       (1, ))
            samp = samp.t()
            sampled = torch.cat(
                [sampled, self.cls_to_val(samp.float())], dim=1)
        return sampled

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def cls_to_val(self, cls: torch.Tensor) -> torch.Tensor:
        return cls.float()


class PixelPredictor(FactoredPredictor):
    def __init__(self, hid_ch: int, n_ch: int = 3):
        super(PixelPredictor, self).__init__(hid_ch, 256, n_ch)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2 - 1

    def cls_to_val(self, cls: torch.Tensor) -> torch.Tensor:
        return cls.float() / 255


class ResBlk(nn.Module):
    @experimental
    def __init__(self, in_ch: int, hid_ch: int, out_ch: int, ks: int,
                 sz: Tuple[int, int]) -> None:
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

    def condition(self, z: torch.Tensor) -> None:
        self.go.condition(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.go(x)
        return x + out


# https://arxiv.org/pdf/1701.05517.pdf
class PixCNNBase(nn.Module):
    @experimental
    def __init__(self,
                 in_ch: int,
                 hid: int,
                 out_ch: int,
                 quant_lvls: int,
                 sz: Tuple[int, int],
                 n_layer: int = 3) -> None:
        super(PixCNNBase, self).__init__()
        self.sz = sz
        self.lin = tnn.CondSeq(
            tnn.TopLeftConv2d(in_ch, hid, 5, center=False, bias=sz),
            nn.ReLU(inplace=True))

        sz2 = sz[0] // 2, sz[1] // 2
        sz4 = sz[0] // 4, sz[1] // 4
        self.l1 = nn.Sequential(
            *[ResBlk(hid, hid * 2, hid, 5, sz) for _ in range(n_layer)])
        self.l2 = nn.Sequential(
            *[ResBlk(hid, hid * 2, hid, 5, sz2) for _ in range(n_layer)])
        self.l3 = nn.Sequential(
            *[ResBlk(hid, hid * 2, hid, 5, sz4) for _ in range(n_layer)])
        self.l4 = nn.Sequential(
            *[ResBlk(hid, hid * 2, hid, 5, sz4) for _ in range(n_layer)])
        self.l4 = nn.Sequential(
            *[ResBlk(hid, hid * 2, hid, 5, sz4) for _ in range(n_layer)])
        self.l5 = nn.Sequential(*[
            ResBlk(hid * 2, hid * 4, hid * 2, 5, sz2) for _ in range(n_layer)
        ])
        self.l6 = nn.Sequential(*[
            ResBlk(hid * 3, hid * 6, hid * 3, 5, sz) for _ in range(n_layer)
        ])

        self.lout = PixelPredictor(hid * 3, out_ch)

    def _body(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)

        x1 = self.l1(x)
        x2 = self.l2(x1[..., ::2, ::2])
        x3 = self.l3(x2[..., ::2, ::2])

        x4 = self.l4(x3)
        x5 = self.l5(torch.cat([F.interpolate(x4, scale_factor=2), x2], dim=1))
        x6 = self.l6(torch.cat([F.interpolate(x5, scale_factor=2), x1], dim=1))
        return F.relu(x6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x6 = self._body(x)
        return self.lout(x6, x)

    def sample_xy(self, x: torch.Tensor, coord_x: int, coord_y: int,
                  temp: float) -> torch.Tensor:
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
    @experimental
    def __init__(self,
                 hid: int,
                 sz: Tuple[int, int],
                 channels: int = 3,
                 n_layer: int = 3) -> None:
        super(PixelCNN, self).__init__(channels,
                                       hid,
                                       channels,
                                       256,
                                       sz,
                                       n_layer=n_layer)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A forward pass for training"""
        return super().forward(x)

    def sample(self, temp: float, N: int) -> torch.Tensor:
        """
        Sample a batch of images

        Args:
            temp (float): the sampling temperature
            N (int): number of images to generate in the batch

        Returns:
            A batch of images
        """
        img = torch.zeros(N,
                          self.channels,
                          *self.sz,
                          device=next(self.parameters()).device).uniform_(
                              0, 1)
        return self.sample_(img, temp)

    def partial_sample(self, x: torch.Tensor, temp: float) -> torch.Tensor:
        x[:, :, x.shape[2] // 2, :] = 0
        return self.sample_(x, temp, start_coord=(x.shape[2] // 2, 0))

    def sample_cond(self, cond: torch.Tensor, temp: float) -> torch.Tensor:
        device = next(self.parameters()).device
        img = torch.empty(cond.shape[0],
                          self.sz[0],
                          *self.sz,
                          device=device).uniform_(0, 1)
        cond_rsz = F.interpolate(cond, size=img.shape[2:], mode='nearest')
        img = torch.cat([img, cond_rsz], dim=1)
        return self.sample_(img, temp)[:, cond.shape[1]:]

    def sample_(self,
                img: torch.Tensor,
                temp: float = 0,
                start_coord: Tuple[int, int] = (0, 0)) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            for row in range(start_coord[0], self.sz[0]):
                for c in range(start_coord[1] if row == start_coord[0] else 0,
                               self.sz[0]):
                    x = self.sample_xy(img * 2 - 1, c, row, temp)
                    img[:, :, row, c] = x

        return img
