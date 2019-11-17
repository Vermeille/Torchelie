import copy
from collections import defaultdict
import os
from pathlib import Path
import torch
from visdom import Visdom

from torchelie.utils import dict_by_key, recursive_state_dict
from torchelie.utils import load_recursive_state_dict, AutoStateDict
from torchelie.metrics.inspector import ClassificationInspector as CIVis
import torchelie.utils as tu

from .avg import *


class WindowedMetricAvg(tu.AutoStateDict):
    def __init__(self, name, post_each_batch=True):
        super(WindowedMetricAvg, self).__init__()
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


class EpochMetricAvg(tu.AutoStateDict):
    def __init__(self, name, post_each_batch=True):
        super(EpochMetricAvg, self).__init__()
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


class AccAvg(tu.AutoStateDict):
    def __init__(self, post_each_batch=True):
        super(AccAvg, self).__init__()
        self.post_each_batch = post_each_batch

    def on_epoch_start(self, state):
        self.avg = RunningAvg()
        if 'acc' in state['metrics']:
            del state['metrics']['acc']

    def on_batch_end(self, state):
        pred, y = state['pred'], state['batch'][1]
        pred = tu.as_multiclass_shape(pred)
        batch_correct = pred.argmax(1).eq(y).float().sum()
        self.avg.log(batch_correct, pred.shape[0])

        if self.post_each_batch:
            state['metrics']['acc'] = self.avg.get()

    def on_epoch_end(self, state):
        state['metrics']['acc'] = self.avg.get()


class MetricsTable(tu.AutoStateDict):
    def __init__(self, post_each_batch=True):
        super(MetricsTable, self).__init__()
        self.post_each_batch = post_each_batch

    def on_epoch_start(self, state):
        if 'table' in state['metrics']:
            del state['metrics']['table']

    def make_html(self, state):
        html = '''
        <style>
        table {
            border: solid 1px #DDEEEE;
            border-collapse: collapse;
            border-spacing: 0;
            font: normal 13px Arial, sans-serif;
        }
        th {
            background-color: #DDEFEF;
            border: solid 1px #DDEEEE;
            color: #336B6B;
            padding: 10px;
            text-align: left;
            text-shadow: 1px 1px 1px #fff;
        }
        td {
            border: solid 1px #DDEEEE;
            color: #333;
            padding: 10px;
            text-shadow: 1px 1px 1px #fff;
        }
        </style>
        <table>
        '''

        for k, v in state['metrics'].items():
            if isinstance(v, float):
                html += '<tr><th>{}</th><td>{}</td></tr>'.format(
                    k, round(v, 6))
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                html += '<tr><th>{}</th><td>{}</td></tr>'.format(
                    k, round(v.item(), 6))
        html += '</table>'
        return html

    def on_batch_end(self, state):
        if self.post_each_batch:
            state['metrics']['table'] = self.make_html(state)

    def on_epoch_end(self, state):
        state['metrics']['table'] = self.make_html(state)


class Optimizer(tu.AutoStateDict):
    def __init__(self, opt, accumulation=1, log_lr=False, log_mom=False):
        super(Optimizer, self).__init__()
        self.opt = opt
        self.accumulation = accumulation
        self.log_lr = log_lr
        self.log_mom = log_mom

    def on_batch_start(self, state):
        if state['iters'] % self.accumulation == 0:
            self.opt.zero_grad()

        if self.log_lr:
            for i in range(len(self.opt.param_groups)):
                pg = self.opt.param_groups[i]
                state['metrics']['lr_' + str(i)] = pg['lr']

        if self.log_mom:
            for i in range(len(self.opt.param_groups)):
                pg = self.opt.param_groups[i]
                if 'momentum' in pg:
                    state['metrics']['mom_' + str(i)] = pg['momentum']
                elif 'betas' in pg:
                    state['metrics']['mom_' + str(i)] = pg['betas'][0]

    def on_batch_end(self, state):
        if (state['iters'] + 1) % self.accumulation == 0:
            self.opt.step()


class LRSched(tu.AutoStateDict):
    def __init__(self, sched, metric='loss', step_each_batch=False):
        super(LRSched, self).__init__()
        self.sched = sched
        self.metric = metric
        self.step_each_batch = step_each_batch

    def state_dict(self):
        if hasattr(self.sched, 'state_dict'):
            return {'scheduler': self.sched.state_dict()}
        return {}

    def load_state_dict(self, dicc):
        if 'scheduler' in dicc:
            self.sched.load_state_dict(dicc['scheduler'])

    def on_batch_end(self, state):
        if self.step_each_batch:
            if self.metric is None:
                self.sched.step()
            else:
                self.sched.step(state['metrics'][self.metric])

    def on_epoch_end(self, state):
        if not self.step_each_batch:
            if self.metric is None:
                self.sched.step()
            else:
                self.sched.step(state['metrics'][self.metric])


class Log(tu.AutoStateDict):
    def __init__(self, from_k, to):
        super(Log, self).__init__()
        self.from_k = from_k
        self.to = to

    def on_batch_end(self, state):
        state['metrics'][self.to] = dict_by_key(state, self.from_k)


class VisdomLogger(tu.AutoStateDict):
    def __init__(self, visdom_env='main', log_every=10, prefix=''):
        super(VisdomLogger, self).__init__(except_names=['vis'])
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
            elif isinstance(x, str):
                self.vis.text(x, win=name, opts=dict(title=name))
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
                assert False, "incorrect type " + x.__class__.__name__


class StdoutLogger(tu.AutoStateDict):
    def __init__(self, log_every=10, prefix=''):
        super(StdoutLogger, self).__init__()
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
            elif isinstance(x, str):
                show[name] = x[:20]
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
        saved = recursive_state_dict(self.objects)
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

        load_recursive_state_dict(loaded, self.objects)

    def on_epoch_end(self, state):
        self.save(state)


class Polyak:
    def __init__(self, original, copy, beta=0.999):
        self.original = original
        self.copy = copy
        self.beta = beta

    @torch.no_grad()
    def on_batch_end(self, state):
        for s, d in zip(self.original.parameters(), self.copy.parameters()):
            d.mul_(self.beta).add_(1 - self.beta, s)


class Counter(tu.AutoStateDict):
    def __init__(self):
        super(Counter, self).__init__()
        self.epoch = -1
        self.iters = -1
        self.epoch_batch = -1

    def on_batch_start(self, state):
        self.iters += 1
        self.epoch_batch += 1

        state['epoch'] = self.epoch
        state['iters'] = self.iters
        state['epoch_batch'] = self.epoch_batch

    def on_epoch_start(self, state):
        self.epoch += 1
        self.epoch_batch = 0

        state['epoch'] = self.epoch
        state['iters'] = self.iters
        state['epoch_batch'] = self.epoch_batch



class ClassificationInspector:
    def __init__(self, nb_show, classes, post_each_batch=True):
        self.vis = CIVis(nb_show, classes, 1. / len(classes))
        self.post_each_batch = post_each_batch

    def on_epoch_start(self, state):
        if 'report' in state['metrics']:
            del state['metrics']['report']
        self.vis.reset()

    def on_batch_end(self, state):
        pred, y, x = state['pred'], state['batch'][1], state['batch'][0]
        self.vis.analyze(x, pred, y)
        if self.post_each_batch:
            state['metrics']['report'] = self.vis.show()

    def on_epoch_end(self, state):
        state['metrics']['report'] = self.vis.show()


class CallDataLoop(tu.AutoStateDict):
    def __init__(self, loop, run_every=100, prefix='test', init_fun=None):
        super(CallDataLoop, self).__init__(except_names=['init_fun'])
        self.loop = loop
        self.run_every = run_every
        self.prefix = prefix
        self.init_fun = init_fun

    def on_batch_end(self, state):
        if state['iters'] % self.run_every == 0:
            if self.init_fun is not None:
                self.init_fun(state)
            out = self.loop.run(1)
            state[self.prefix + '_metrics'] = copy.deepcopy(out['metrics'])
