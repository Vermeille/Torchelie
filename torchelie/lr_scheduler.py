import math
import torchelie.utils as tu
from typing import List, Tuple


class CurriculumScheduler:
    """
    Allow to pre-specify learning rate and momentum changes

    Args:
        optimizer (torch.optim.Optimizer): the optimizer to schedule. Currently
            works only with SGD
        schedule (list): a schedule. It's a list of keypoints where each
            element is a 3-tuple like (iteration number, lr, mom). Values are
            interpolated linearly between neighboring keypoints
        last_iter (int): starting iteration
    """
    def __init__(self,
                 optimizer,
                 schedule: List[Tuple[float, float, float]],
                 last_iter: int = -1):
        self.optimizer = optimizer
        self.schedule = schedule
        self.last_iter = last_iter

    def state_dict(self):
        return {'last_iter': self.last_iter}

    def load_state_dict(self, state):
        self.last_iter = state['last_iter']

    def step(self, *unused) -> None:
        """
        Step the scheduler to another iteration
        """
        self.last_iter += 1
        the_lr = self.schedule[-1][1]
        the_mom = self.schedule[-1][2]
        for lo, hi in zip(self.schedule[:-1], self.schedule[1:]):
            limit_lo, lr_lo, mom_lo = lo
            lim_hi, lr_hi, mom_hi = hi

            if limit_lo <= self.last_iter < lim_hi:
                t = tu.ilerp(limit_lo, lim_hi, self.last_iter)
                the_lr = tu.lerp(lr_lo, lr_hi, t)
                the_mom = tu.lerp(mom_lo, mom_hi, t)

        for group in self.optimizer.param_groups:
            group['lr'] = the_lr
            if 'momentum' in group:
                group['momentum'] = the_mom
            elif 'betas' in group:
                group['betas'] = (the_mom, group['betas'][1])

    def __repr__(self):
        return "CurriculumScheduler({})".format(self.schedule)


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
