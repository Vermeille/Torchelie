class CurriculumScheduler:
    """
    Allow to pre-specify learning rate and momentum changes

    Args:
        optimizer (torch.optim.Optimizer): the optimizer to schedule
        schedule (list): a schedule. It's a list of keypoints where each
            element is a 3-tuple like (iteration number, lr, mom). Values are
            interpolated linearly between neighboring keypoints
        last_iter (int): starting iteration
    """
    def __init__(self, optimizer, schedule, last_iter=-1):
        print(type(schedule), schedule)
        self.optimizer = optimizer
        self.schedule = schedule
        self.last_iter = last_iter

    def step(self, *unused):
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
                t = mathutils.get_t(limit_lo, lim_hi, self.last_iter)
                the_lr = mathutils.lerp(lr_lo, lr_hi, t)
                the_mom = mathutils.lerp(mom_lo, mom_hi, t)

        for group in self.optimizer.param_groups:
            group['lr'] = the_lr
            group['momentum'] = the_mom

    def __repr__(self):
        return "CurriculumScheduler({})".format(self.schedule)
