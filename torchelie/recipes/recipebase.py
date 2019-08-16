import torch

from visdom import Visdom


class RecipeBase:
    def __init__(self, visdom_env=None, log_every=10):
        self.vis = None
        self.iters = 0
        self.log_every = log_every
        if visdom_env is not None:
            self.vis = Visdom(env=visdom_env)
            self.vis.close()

    def log(self, xs, store_history=[]):
        if self.vis is None or self.iters % self.log_every != 0:
            return

        for name, x in xs.items():
            if isinstance(x, (float, int)):
                self.vis.line(X=[self.iters],
                              Y=[x],
                              update='append',
                              win=name,
                              opts=dict(title=name))
            elif isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    self.vis.line(X=[self.iters],
                                  Y=[x.item()],
                                  update='append',
                                  win=name,
                                  opts=dict(title=name))
                elif x.dim() == 2:
                    self.vis.heatmap(x, win=name, opts=dict(title=name))
                elif x.dim() == 3:
                    self.vis.image(x, win=name, opts=dict(title=name,
                        store_history=name in store_history))
                elif x.dim() == 4:
                    self.vis.images(x, win=name, opts=dict(title=name,
                        store_history=name in store_history))
                else:
                    assert False, "incorrect tensor dim"
            else:
                assert False, "incorrect tensor dim"
