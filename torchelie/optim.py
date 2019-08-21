import math
import torch
from torch.optim import Optimizer


class DeepDreamOptim(Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(DeepDreamOptim, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]

                step_size = group['lr']

                p.data.addcdiv_(-step_size, p.grad.data, p.grad.data.abs().mean())

        return loss

