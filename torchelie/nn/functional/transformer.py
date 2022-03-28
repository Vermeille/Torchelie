import math
import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F


def local_attention_2d(x: Tensor, conv_kqv: nn.Conv2d, posenc: Tensor,
                       num_heads: int, patch_size: int) -> Tensor:
    B, fullC, fullH, fullW = x.shape
    N = num_heads
    P = patch_size
    C, H, W = fullC // N, fullH // P, fullW // P

    x = x.view(B, N, C, H, P, W, P).permute(0, 3, 5, 1, 2, 4,
                                            6).reshape(B * H * W, N * C, P, P)
    x += posenc
    k, q, v = torch.chunk(F.conv2d(x, conv_kqv.weight / math.sqrt(C),
                                   conv_kqv.bias),
                          3,
                          dim=1)
    k = k.view(B, H * W, N, C, P * P)
    q = q.view(B, H * W, N, C, P * P)

    kq = torch.softmax(torch.matmul(q.transpose(-1, -2), k), dim=-1)
    v = v.view(B, H * W, N, C, P * P)
    kqv = torch.matmul(v, kq.transpose(-1, -2)).view(B, H, W, N, C, P, P)
    kqv = kqv.permute(0, 3, 4, 1, 5, 2, 6).reshape(B, fullC, fullH, fullW)
    return kqv
