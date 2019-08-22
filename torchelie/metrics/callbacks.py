import torch
from visdom import Visdom

from torchelie.utils import dict_by_key

from .avg import *


class WindowedMetricAvg:
    def __init__(self, name, post_each_batch=True):
        self.name = name
        self.avg = WindowAvg(k=100)
        self.post_each_batch = post_each_batch

    def on_epoch_start(self, state):
        if self.name in state['metrics']:
            del state['metrics'][self.name]

    def on_batch_end(self, state):
        self.avg.log(state[self.name])
        if self.post_each_batch:
            state['metrics'][self.name] = self.avg.get()

    def on_epoch_end(self, state):
        state['metrics'][self.name] = self.avg.get()


class EpochMetricAvg:
    def __init__(self, name, post_each_batch=True):
        self.name = name
        self.post_each_batch = post_each_batch

    def on_epoch_start(self, state):
        self.avg = RunningAvg()
        if self.name in state['metrics']:
            del state['metrics'][self.name]

    def on_batch_end(self, state):
        self.avg.log(state[self.name])
        if self.post_each_batch:
            state['metrics'][self.name] = self.avg.get()

    def on_epoch_end(self, state):
        state['metrics'][self.name] = self.avg.get()


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


class Checkpoint:
    """FIXME: WIP"""
    def __init__(self, filename_base, keys):
        self.filename_base = filename_base
        self.keys = keys
        self.nb_saved = 0

    def save(self, state):
        saved = {}
        for k in self.keys:
            m = state[k]
            if hasattr(m, 'state_dict'):
                saved[k] = m.state_dict()
            else:
                saved[k] = m
        torch.save(saved, self.filename())
        self.nb_saved += 1

    def filename(self):
        return self.filename_base + '_' + str(self.nb_saved) + '.pth'

    def load(self, state):
        while True:
            try:
                loaded = torch.load(self.filename())
            except:
                pass
            self.nb_saved += 1

        for k in self.keys:
            m = state[k]
            if hasattr(m, 'load_state_dict'):
                m.load_state_dict(loaded[k])
            else:
                state[k] = loaded[k]

    def on_epoch_end(self, state):
        self.save(state)
