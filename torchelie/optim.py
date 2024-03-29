import math
import torch
from torch.optim import Optimizer
from collections import defaultdict


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

                step_size = group['lr']
                eps = group['eps']

                p.data.add_(p.grad.data,
                            alpha=-step_size / (eps + p.grad.data.abs().mean()))

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
        defaults = {'lr': lr, 'beta': beta, 'weight_decay': weight_decay}
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
                mom.mul_(beta).add_(grad, alpha=1 - beta)

                s = torch.sign(grad).mul_(torch.sign(mom)).add_(1)

                p.data.addcmul_(-lr * s, grad)
                p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss


class AdaBelief(Optimizer):
    r"""Implements AdaBelief algorithm.

    AdaBelief from `AdaBelief Optimizer: Adapting Stepsizes by the Belief in
    Observed Gradients <https://arxiv.org/abs/2010.07468>`_

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(
                betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdaBelief, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps = group['eps']
                lr = group['lr']

                state['step'] += 1
                t = state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                belief = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(belief, belief, value=1 -
                        beta2).add_(eps)
                # This is everything. See EAdam.

                var = exp_avg_sq.mul(1 / (1 - beta2**t))
                var.sqrt_()

                # Perform stepweight decay
                p.data.mul_(1 - lr * group['weight_decay'])

                p.data.addcdiv_(exp_avg, var, value=-lr / (1 - beta1**t))

        return loss


class RAdamW(Optimizer):
    r"""Implements RAdamW algorithm.

    RAdam from `On the Variance of the Adaptive Learning Rate and Beyond
    <https://arxiv.org/abs/1908.03265v1>`_

    * `Adam: A Method for Stochastic Optimization
      <https://arxiv.org/abs/1412.6980>`_
    * `Decoupled Weight Decay Regularization
      <https://arxiv.org/abs/1711.05101>`_
    * `On the Convergence of Adam and Beyond
      <https://openreview.net/forum?id=ryQu7f-RZ>`_
    * `On the Variance of the Adaptive Learning Rate and Beyond
      <https://arxiv.org/abs/1908.03265v1>`_

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(
                betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps = group['eps']
                lr = group['lr']
                if 'rho_inf' not in group:
                    group['rho_inf'] = 2 / (1 - beta2) - 1
                rho_inf = group['rho_inf']

                state['step'] += 1
                t = state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                rho_t = rho_inf - ((2 * t * (beta2**t)) / (1 - beta2**t))

                # Perform stepweight decay
                p.data.mul_(1 - lr * group['weight_decay'])

                if rho_t >= 5:
                    var = exp_avg_sq.sqrt().add_(eps)
                    r = math.sqrt(
                        (1 - beta2**t) * ((rho_t - 4) * (rho_t - 2) * rho_inf) /
                        ((rho_inf - 4) * (rho_inf - 2) * rho_t))

                    p.data.addcdiv_(exp_avg,
                                    var,
                                    value=-lr * r / (1 - beta1**t))
                else:
                    p.data.add_(exp_avg, alpha=-lr / (1 - beta1**t))

        return loss


class Lookahead(Optimizer):
    """
    Implements Lookahead from `Lookahead Optimizer: k steps forward, 1 step
    back` (Zhang et al, 2019)

    Args:
        base_optimizer (Optimizer): an optimizer for the inner loop
        alpha (float): outer loop learning rate
        k (int): number of steps in the inner loop
    """

    def __init__(self, base_optimizer, alpha=0.5, k=5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Invalid slow update rate: ' + str(alpha))
        if not 1 <= k:
            raise ValueError('Invalid lookahead steps: ' + str(k))

        pgroups = [{'params': p['params']} for p in base_optimizer.param_groups]
        super().__init__(pgroups, dict(la_alpha=alpha, la_k=k))
        self.optimizer = base_optimizer

        with torch.no_grad():
            for group in pgroups:
                for p in group['params']:
                    self.state[p]['la_slow'] = p.clone()

    def state_dict(self):
        return {
            'lookahead': super().state_dict(),
            'opt': self.optimizer.state_dict()
        }

    def load_state_dict(self, state):
        super().load_state_dict(state['lookahead'])
        self.optimizer.load_state_dict(state['opt'])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()

        for group in self.param_groups:
            k = group['la_k']
            alpha = group['la_alpha']

            for p in group['params']:
                state = self.state[p]

                steps = state.get('la_steps', 0)
                state['la_steps'] = steps + 1

                if steps % k != 0:
                    continue

                q = state['la_slow']
                # q = q + alpha (p - q)
                # q = q + alpha p - alpha q
                # q = alpha p + (1 - alpha) q
                q.data.add_(p.data - q.data, alpha=alpha)
                p.data.copy_(q.data)
        return loss


class Lamb(Optimizer):
    r"""Implements Lamb algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        clamp_value: clamp weight_norm in (0,clamp_value) (default: 10)
            set to a high value to avoid it (e.g 10e3)
        adam: always use trust ratio = 1, which turns this
            into Adam. Useful for comparison purposes. (default: False)
        debias: debias adam by (1 - beta**step) (default: False)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Lamb(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1904.00962

    Note:
        Reference code: https://github.com/cybertronai/pytorch-lamb
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: (float, float) = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        clamp_value: float = 10,
        adam: bool = False,
        debias: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(
                betas[1]))
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay))
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias

        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = ('Lamb does not support sparse gradients, '
                           'please consider SparseAdam instead')
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                if self.debias:
                    bias_correction = math.sqrt(1 - beta2**state['step'])
                    bias_correction /= 1 - beta1**state['step']
                else:
                    bias_correction = 1

                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] * bias_correction

                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss
