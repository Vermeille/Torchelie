import torch
import math
import torchelie.utils as tu
from typing import List, Tuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CurriculumScheduler(_LRScheduler):
    """
    Allow to pre-specify learning rate and momentum changes

    Args:
        optimizer (torch.optim.Optimizer): the optimizer to schedule. Currently
            works only with SGD
        schedule (list): a schedule. It's a list of keypoints where each
            element is a 3-tuple like (iteration number, lr multiplier, mom).
            Values are interpolated linearly between neighboring keypoints
        last_epoch (int): starting iteration
    """

    def __init__(self,
                 optimizer,
                 schedule: List[Tuple[float, float, float]],
                 last_epoch: int = -1,
                 verbose: bool = False,
                 transition: str = 'linear'):
        self.schedule = schedule
        self.transition = transition
        if transition == 'log':
            self.schedule = [(a, math.log(b), math.log(c))
                             for a, b, c in self.schedule]
        super().__init__(optimizer, last_epoch, verbose)

    def step(self, *unused) -> None:
        """
        Step the scheduler to another iteration
        """
        self.last_epoch += 1
        lr_mul = self.schedule[-1][1]
        the_mom = self.schedule[-1][2]
        t = 1
        for lo, hi in zip(self.schedule[:-1], self.schedule[1:]):
            limit_lo, lr_lo, mom_lo = lo
            lim_hi, lr_hi, mom_hi = hi

            if limit_lo <= self.last_epoch < lim_hi:
                t = tu.ilerp(limit_lo, lim_hi, self.last_epoch)
                lr_mul = tu.lerp(lr_lo, lr_hi, t)
                the_mom = None
                if not (mom_lo is None or mom_hi is None):
                    the_mom = tu.lerp(mom_lo, mom_hi, t)
                break

        if self.transition == 'log':
            lr_mul = math.exp(lr_mul)
            if the_mom is not None:
                the_mom = math.exp(the_mom)

        elif self.transition == 'cosine':
            cos = math.cos(t * math.pi + math.pi) / 2 + 0.5
            lr_mul = (lr_hi - lr_lo) * cos + lr_lo
            if the_mom is not None:
                the_mom = (mom_hi - mom_lo) * cos + mom_lo

        for group in self.optimizer.param_groups:
            group['lr'] = lr_mul * group['initial_lr']
            if the_mom is not None:
                if 'momentum' in group:
                    group['momentum'] = the_mom
                elif 'betas' in group:
                    group['betas'] = (the_mom, group['betas'][1])

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.schedule)


class LinearDecay(CurriculumScheduler):

    def __init__(self,
                 optimizer,
                 total_iters: int,
                 warmup_ratio: float = 0.05,
                 last_epoch: int = -1,
                 verbose: bool = False):
        if warmup_ratio == 0:
            sched = [(0, 1, None), (total_iters, 0, None)]
        else:
            sched = [(0, 0, None), (int(total_iters * warmup_ratio), 1, None),
                     (total_iters, 0, None)]
        super().__init__(optimizer, sched, last_epoch, verbose)


class CosineDecay(_LRScheduler):
    """
    Allow to pre-specify learning rate and momentum changes

    Args:
        optimizer (torch.optim.Optimizer): the optimizer to schedule. Currently
            works only with SGD
        schedule (list): a schedule. It's a list of keypoints where each
            element is a 3-tuple like (iteration number, lr multiplier, mom).
            Values are interpolated linearly between neighboring keypoints
        last_epoch (int): starting iteration
    """

    def __init__(self,
                 optimizer,
                 total_iters: int,
                 warmup_ratio: float = 0.05,
                 last_epoch: int = -1,
                 verbose: bool = False):
        self.total_iters = total_iters
        self.warmup = warmup_ratio
        super().__init__(optimizer, last_epoch, verbose)

    def step(self, *unused) -> None:
        """
        Step the scheduler to another iteration
        """
        self.last_epoch += 1

        warmup_iters = int(self.warmup * self.total_iters)

        if self.last_epoch < warmup_iters:
            warmup = min(self.last_epoch / warmup_iters, 1)
            lr_mul = 1 - 0.5 * (math.cos(warmup * math.pi) + 1)
        else:
            progress = (self.last_epoch - warmup_iters) / (self.total_iters -
                                                           warmup_iters)
            lr_mul = 0.5 * (math.cos(progress * math.pi) + 1)

        for group in self.optimizer.param_groups:
            group['lr'] = lr_mul * group['initial_lr']

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.total_iters)


class OneCycle(CurriculumScheduler):
    """
    Implements 1cycle policy.

    Goes from:
    - `lr[0]` to `lr[1]` during `num_iters // 3` iterations
    - `lr[1]` to `lr[0]` during another `num_iters // 3` iterations
    - `lr[0]` to `lr[0] // 10` during another `num_iters // 3` iterations

    Args:
        opt (Optimizer): the optimizer on which to modulate lr
        lr (2-tuple): lr range
        num_iters (int): total number of iterations
        mom (2-tuple): momentum range
        last_iter (int): last_iteration index
    """

    def __init__(self,
                 opt,
                 lr: Tuple[float, float],
                 num_iters: int,
                 mom: Tuple[float, float] = (0.95, 0.85),
                 log: bool = False,
                 last_iter: int = -1):
        self.log = log

        if log:
            lr = math.log(lr[0]), math.log(lr[1])
            mom = math.log(mom[0]), math.log(mom[1])

        third = num_iters // 3
        super(OneCycle, self).__init__(
            opt, [(0, lr[0], mom[0]), (third, lr[1], mom[1]),
                  (2 * third, lr[0], mom[0]),
                  (num_iters, lr[0] / (lr[1] / lr[0] * 10000), mom[0])],
            last_iter=last_iter)

    def step(self, *unused):
        super(OneCycle, self).step()

        if not self.log:
            return

        for group in self.optimizer.param_groups:
            group['lr'] = math.exp(group['lr'])
            if 'momentum' in group:
                group['momentum'] = math.exp(group['momentum'])
            elif 'betas' in group:
                group['betas'] = (math.exp(group['betas'][0]),
                                  group['betas'][1])

    def __repr__(self):
        return 'OneCycle({})'.format(self.schedule)


class WSD(CurriculumScheduler):
    """
    Implements the Warmup-Step-Decay schedule.

    Goes from:
    - 0 to 1 during `warmup_iters` iterations
    - stays at 1 during `step_iters` iterations
    - goes from 1 to `end_mul` during `decay_iters` iterations

    Args:
        opt (Optimizer): the optimizer on which to modulate lr
        warmup_iters (int): number of iterations to warmup
        step_iters (int): number of iterations to keep the same lr
        decay_iters (optional, int): number of iterations to decay lr. Defaults
            to 10% of step_iters.
        end_mul (optional, float): final lr multiplier (default to 0.1)
        last_iter (int): last_iteration index
    """

    def __init__(self,
                 opt,
                 warmup_iters: int,
                 step_iters: int,
                 decay_iters: int = None,
                 end_mul: float = 0.1,
                 last_iter: int = -1):
        if decay_iters is None:
            decay_iters = step_iters // 10

        super(WSD, self).__init__(opt, [(0, 0, None), (warmup_iters, 1, None),
                                        (step_iters - decay_iters, 1, None),
                                        (step_iters, end_mul, None)],
                                  last_epoch=last_iter)

    def __repr__(self):
        return 'WSD({})'.format(self.schedule)


class HyperbolicTangentDecay(_LRScheduler):
    """
    Coming from Stochastic gradient descent with hyperbolic-tangent decay on
    classification (Hsueh et al., 2019), keeps a flat LR for about 70% of the
    training then decays following a hypertangent curve.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 n_iters_total: int,
                 tanh_lower_bound: int = -6,
                 tanh_upper_bound: int = 3,
                 last_epoch: int = -1,
                 verbose: bool = False):
        self.n_iters_total = n_iters_total
        self.tanh_lower_bound = tanh_lower_bound
        self.tanh_upper_bound = tanh_upper_bound
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        t = self.last_epoch / self.n_iters_total
        L = self.tanh_lower_bound
        U = self.tanh_upper_bound
        return [
            b_lr * 0.5 * (1 - math.tanh(L + (U - L) * t))
            for b_lr in self.base_lrs
        ]

    def __repr__(self) -> str:
        return 'HyperbolicTangentDecay({})'.format(
            tu.indent("\n".join([
                '{}={}'.format(k, v)
                for k, v in self.__dict__.items()
                if k != 'optimizer'
            ])))
