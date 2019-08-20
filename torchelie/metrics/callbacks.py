import torch

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

