import math
import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F


def local_attention_2d(x: Tensor, conv_kqv: nn.Conv2d, posenc: Tensor,
                       num_heads: int, patch_size: int) -> Tensor:
    B, inC, fullH, fullW = x.shape
    N = num_heads
    P = patch_size
    H, W = fullH // P, fullW // P

    x = x.view(B, inC, H, P, W, P).permute(0, 2, 4, 1, 3,
                                           5).reshape(B * H * W, inC, P, P)
    k, q, v = torch.chunk(F.conv2d(x, conv_kqv.weight / math.sqrt(inC // N),
                                   conv_kqv.bias),
                          3,
                          dim=1)
    hidC = k.shape[1] // N
    k = k.view(B, H * W, N, hidC, P * P)
    q = q.view(B, H * W, N, hidC, P * P)

    kq = torch.softmax(torch.matmul(q.transpose(-1, -2), k) + posenc, dim=-1)
    v = v.view(B, H * W, N, hidC, P * P)
    kqv = torch.matmul(v, kq.transpose(-1, -2)).view(B, H, W, N, hidC, P, P)
    kqv = kqv.permute(0, 3, 4, 1, 5, 2, 6).reshape(B, hidC * N, fullH, fullW)
    return kqv
