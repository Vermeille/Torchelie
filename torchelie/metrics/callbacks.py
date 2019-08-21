import torch
from visdom import Visdom

from .avg import *


class WindowedLossAvg:
    def __init__(self, post_each_batch=True):
        self.avg = WindowAvg(k=100)
        self.post_each_batch = post_each_batch

    def on_epoch_start(self, state):
        if 'loss' in state['metrics']:
            del state['metrics']['loss']

    def on_batch_end(self, state):
        self.avg.log(state['loss'])
        if self.post_each_batch:
            state['metrics']['loss'] = self.avg.get()

    def on_epoch_end(self, state):
        state['metrics']['loss'] = self.avg.get()


class EpochLossAvg:
    def __init__(self, post_each_batch=True):
        self.post_each_batch = post_each_batch

    def on_epoch_start(self, state):
        self.avg = RunningAvg()
        if 'loss' in state['metrics']:
            del state['metrics']['loss']

    def on_batch_end(self, state):
        self.avg.log(state['loss'])
        if self.post_each_batch:
            state['metrics']['loss'] = self.avg.get()

    def on_epoch_end(self, state):
        state['metrics']['loss'] = self.avg.get()


class AccAvg:
    def __init__(self, post_each_batch=True):
        self.post_each_batch = post_each_batch

    def on_epoch_start(self, state):
        self.avg = RunningAvg()
        if 'acc' in state['metrics']:
            del state['metrics']['acc']

    def on_batch_end(self, state):
        pred, y = state['pred'], state['batch'][1]
        batch_correct = pred.argmax(1).eq(y).float().sum()
        self.avg.log(batch_correct, pred.shape[0])

        if self.post_each_batch:
            state['metrics']['acc'] = self.avg.get()

    def on_epoch_end(self, state):
        state['metrics']['acc'] = self.avg.get()


class VisdomLogger:
    def __init__(self, visdom_env='main', log_every=10, prefix=''):
        self.vis = None
        self.iters = 0
        self.epoch = 0
        self.epoch_iters = 0
        self.log_every = log_every
        self.prefix = prefix
        if visdom_env is not None:
            self.vis = Visdom(env=visdom_env)
            self.vis.close()

    def on_batch_end(self, state):
        if self.log_every != -1 and self.iters % self.log_every == 0:
            self.log(state['metrics'])
        self.iters += 1
        self.epoch_iters += 1

    def on_epoch_end(self, state):
        self.log(state['metrics'])
        self.epoch += 1
        self.epoch_iters = 0

    def log(self, xs, store_history=[]):
        for name, x in xs.items():
            name = self.prefix + name
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
                    self.vis.image(x,
                                   win=name,
                                   opts=dict(
                                       title=name,
                                       store_history=name in store_history))
                elif x.dim() == 4:
                    self.vis.images(x,
                                    win=name,
                                    opts=dict(
                                        title=name,
                                        store_history=name in store_history))
                else:
                    assert False, "incorrect tensor dim"
            else:
                assert False, "incorrect tensor dim"


class StdoutLogger:
    def __init__(self, log_every=10, prefix=''):
        self.vis = None
        self.iters = 0
        self.epoch = 0
        self.epoch_iters = 0
        self.log_every = log_every
        self.prefix = prefix

    def on_batch_end(self, state):
        if self.log_every != -1 and self.iters % self.log_every == 0:
            self.log(state['metrics'])
        self.iters += 1
        self.epoch_iters += 1

    def on_epoch_end(self, state):
        self.log(state['metrics'])
        self.epoch += 1
        self.epoch_iters = 0

    def log(self, xs, store_history=[]):
        show = {}
        for name, x in xs.items():
            if isinstance(x, (float, int)):
                show[name] = "{:.4f}".format(x)
            elif isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    show[name] = "{:.4f}".format(x.item())
                elif x.dim() <= 4:
                    pass
                else:
                    assert False, "incorrect tensor dim"
            else:
                assert False, "incorrect tensor dim"
        print(self.prefix, '| Ep.', self.epoch, 'It', self.epoch_iters, '|',
              show)


class LogInput:
    def on_batch_end(self, state):
        state['metrics']['x'] = state['x']
