import math
import torch
from torch.optim import Optimizer


class DeepDreamOptim(Optimizer):
    r"""Optimizer used by Deep Dream. It rescales the gradient by the average of
    the absolute values of the gradient.

    :math:`\theta_i := \theta_i - lr \frac{g_i}{\epsilon+\frac{1}{M}\sum_j^M |g_j|}`

    Args:
        params: parameters as expected by Pytorch's optimizers
        lr (float): the learning rate
        eps (float): epsilon value to avoid dividing by zero
    """

    def __init__(self, params, lr=1e-3, eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(DeepDreamOptim, self).__init__(params, defaults)

    def step(self, closure=None):
        """Update the weights

        Args:
            closure (optional fn): a function that computes gradients
        """
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
                eps = group['eps']

                p.data.addcdiv_(-step_size, p.grad.data,
                                eps + p.grad.data.abs().mean())

        return loss


class AddSign(Optimizer):
    r"""AddSign optimizer from Neural Optimiser search with Reinforcment
    learning (Bello et al, 2017)

    :math:`\theta_i := \theta_i - \text{lr}(1+\text{sign}(g)*\text{sign}(m))*g`

    Args:
        params: parameters as expected by Pytorch's optimizers
        lr (float): the learning rate
        eps (float): epsilon value to avoid dividing by zero
    """
    def __init__(self, params, lr=1e-3, beta=0.9, weight_decay=0):
        defaults = {
                'lr': lr,
                'beta': beta,
                'weight_decay': weight_decay
        }
        super(AddSign, self).__init__(params, defaults)


    def step(self, closure=None):
        """Update the weights

        Args:
            closure (optional fn): a function that computes gradients
        """
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['mom'] = torch.zeros_like(p.data)

                lr = group['lr']
                beta = group['beta']
                weight_decay = group['weight_decay']

                state['step'] += 1
                mom = state['mom']
                mom.mul_(beta).add_(1 - beta, grad)

                s = torch.sign(grad).mul_(torch.sign(mom)).add_(1)

                p.data.addcmul_(-lr * s, grad)
                p.data.add_(-lr * weight_decay, p.data)

        return loss

