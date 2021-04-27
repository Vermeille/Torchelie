import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Constraint(nn.Module):
    """
    From `Ranjan 2017`_ , L2-constrained Softmax Loss for Discriminative Face
    Verification.

    Args:
        dim (int): number of channels of the feature vector
        num_classes (int): number of identities
        fixed (bool): whether to use the fixed or dynamic version of AdaCos
            (default: False)

    :: _Ranjan 2017: https://arxiv.org/abs/1703.09507
    """
    def __init__(self, dim: int, num_classes: int, s: float = 30.):
        super(L2Constraint, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, dim))
        nn.init.orthogonal_(self.weight)
        self.num_classes = num_classes
        self.s = s

    def forward(self, input: torch.Tensor,
                label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            input (tensor): feature vectors
            label (tensor): labels

        Returns:
            scaled cosine logits, cosine logits
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        output = cosine * self.s

        return output, cosine

    def __repr__(self):
        return "L2CrossEntropy(s={})".format(self.s)


class AdaCos(nn.Module):
    """
    From AdaCos: [Adaptively Scaling Cosine Logits for Effectively Learning
    Deep Face Representations](https://arxiv.org/abs/1905.00292)

    Args:
        dim (int): number of channels of the feature vector
        num_classes (int): number of identities
        fixed (bool): whether to use the fixed or dynamic version of AdaCos
            (default: False)
        estimate_B (bool): is using dynamic AdaCos, B is estimated from the
            real angles of the cosine similarity. However I found that this
            method was not numerically stable and experimented with the
            approximation :code:`B = num_classes - 1` that was more satisfying.
    """
    s: torch.Tensor

    def __init__(self,
                 dim: int,
                 num_classes: int,
                 fixed: bool = False,
                 estimate_B: bool = False):
        super(AdaCos, self).__init__()
        self.fixed = fixed
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, dim))
        nn.init.xavier_normal_(self.weight)
        self.num_classes = num_classes
        self.register_buffer(
            's', torch.tensor(math.sqrt(2) * math.log(num_classes - 1)))
        self.register_buffer('B', torch.tensor(num_classes - 1.))
        self.estimate_B = estimate_B

    def forward(
        self, input: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            input (tensor): feature vectors
            label (tensor): labels

        Returns:
            scaled cosine logits, cosine logits
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if self.fixed:
            return cosine * self.s, cosine

        if self.training:
            with torch.no_grad():
                correct_cos = torch.gather(cosine, 1, label.unsqueeze(1))
                theta_med = torch.acos(correct_cos).median()

                if self.estimate_B:
                    output = cosine * self.s
                    expout = output.exp()
                    correct_expout = torch.gather(expout, 1,
                                                  label.unsqueeze(1))
                    correct_expout.squeeze_()
                    self.B = torch.mean(expout.sum(1) - correct_expout, dim=0)

                self.s = torch.log(self.B) / math.cos(
                    min(math.pi / 4, theta_med.item()))
        return cosine * self.s, cosine

    def __repr__(self) -> str:
        return "FixedAdaCos(fixed={})".format(self.fixed)
