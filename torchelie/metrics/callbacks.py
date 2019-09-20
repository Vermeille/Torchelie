import os
from pathlib import Path
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


class Log:
    def __init__(self, from_k, to):
        self.from_k = from_k
        self.to = to

    def on_batch_end(self, state):
        state['metrics'][self.to] = dict_by_key(state, self.from_k)


class VisdomLogger:
    def __init__(self, visdom_env='main', log_every=10, prefix=''):
        self.vis = None
        self.log_every = log_every
        self.prefix = prefix
        if visdom_env is not None:
            self.vis = Visdom(env=visdom_env)
            self.vis.close()

    def on_batch_end(self, state):
        iters = state['iters']
        if self.log_every != -1 and iters % self.log_every == 0:
            self.log(iters, state['metrics'])

    def on_epoch_end(self, state):
        self.log(state['iters'], state['metrics'])

    def log(self, iters, xs, store_history=[]):
        if self.vis is None:
            return

        for name, x in xs.items():
            name = self.prefix + name
            if isinstance(x, (float, int)):
                self.vis.line(X=[iters],
                              Y=[x],
                              update='append',
                              win=name,
                              opts=dict(title=name),
                              name=name)
            elif isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    self.vis.line(X=[iters],
                                  Y=[x.item()],
                                  update='append',
                                  win=name,
                                  opts=dict(title=name),
                                  name=name)
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
        self.log_every = log_every
        self.prefix = prefix

    def on_batch_end(self, state):
        iters = state['iters']
        if self.log_every != -1 and iters % self.log_every == 0:
            self.log(state['metrics'], state['epoch'], state['epoch_batch'])

    def on_epoch_end(self, state):
        self.log(state['metrics'], state['epoch'], state['epoch_batch'])

    def log(self, xs, epoch, epoch_batch, store_history=[]):
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
        print(self.prefix, '| Ep.', epoch, 'It', epoch_batch, '|', show)


class Checkpoint:
    """FIXME: WIP"""
    def __init__(self, filename_base, objects):
        self.filename_base = filename_base
        self.objects = objects
        self.nb_saved = 0

    def save(self, state):
        saved = {}
        for k, m in self.objects.items():
            if hasattr(m, 'state_dict'):
                saved[k] = m.state_dict()
            else:
                saved[k] = m
        try:
            Path(self.filename()).parent.mkdir()
        except:
            pass
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

class CallbacksRunner:
    def __init__(self, cbs):
        self.cbs = cbs

    def __call__(self, name, *args, **kwargs):
        for cb in self.cbs:
            if hasattr(cb, name):
                getattr(cb, name)(*args, **kwargs)
