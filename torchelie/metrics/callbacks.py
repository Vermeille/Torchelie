"""
Those are callbacks that can be attached to a recipe. They read a write from a
shared state.

WARNING: this might move to torchelie.recipes.callbacks
"""
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
    """
    Log to the metrics a window averaged value in the current state

    Args:
        name (str): the name of the value to log
        post_each_batch (bool): whether to post on each batch (True, default),
            or only on epoch ends (False)
    """

    def __init__(self, name, post_each_batch=True):
        super(WindowedMetricAvg, self).__init__()
        self.name = name
        self.avg = WindowAvg(k=100)
        self.post_each_batch = post_each_batch

    def on_epoch_start(self, state):
        if self.name in state['metrics']:
            del state['metrics'][self.name]

    @torch.no_grad()
    def on_batch_end(self, state):
        self.avg.log(state[self.name])
        if self.post_each_batch:
            state['metrics'][self.name] = self.avg.get()

    def on_epoch_end(self, state):
        state['metrics'][self.name] = self.avg.get()


class EpochMetricAvg(tu.AutoStateDict):
    """
    Log to the metrics a value averaged over an epoch in the current state

    Args:
        name (str): the name of the value to log
        post_each_batch (bool): whether to post on each batch (True, default),
            or only on epoch ends (False). Notice that posting on each batch
            necessarily yields an approximate average.
    """

    def __init__(self, name, post_each_batch=True):
        super(EpochMetricAvg, self).__init__()
        self.name = name
        self.post_each_batch = post_each_batch

    def on_epoch_start(self, state):
        self.avg = RunningAvg()
        if self.name in state['metrics']:
            del state['metrics'][self.name]

    @torch.no_grad()
    def on_batch_end(self, state):
        self.avg.log(state[self.name])
        if self.post_each_batch:
            state['metrics'][self.name] = self.avg.get()

    def on_epoch_end(self, state):
        state['metrics'][self.name] = self.avg.get()


class AccAvg(tu.AutoStateDict):
    """
    Log the average accuracy to the metrics. The true classes is expected in
    :code:`state['batch'][1]`, and the logits predictions are expected in
    :code:`state['preds']`.

    Args:
        post_each_batch (bool): whether to post on each batch or on epoch end.
            Default: True.
    """

    def __init__(self, post_each_batch=True, avg_type='running'):
        super(AccAvg, self).__init__()
        self.post_each_batch = post_each_batch
        avg = {
            'running': RunningAvg,
            'moving': ExponentialAvg,
            'window': WindowAvg
        }
        self.avg = avg[avg_type]()

    def on_epoch_start(self, state):
        if isinstance(self.avg, RunningAvg):
            self.avg = RunningAvg()

        if 'acc' in state['metrics']:
            del state['metrics']['acc']

    @torch.no_grad()
    def on_batch_end(self, state):
        pred, y = state['pred'], state['batch'][1]
        pred = tu.as_multiclass_shape(pred)
        batch_correct = pred.argmax(1).eq(y).float()
        if isinstance(self.avg, RunningAvg):
            self.avg.log(batch_correct.sum(), pred.shape[0])
        else:
            self.avg.log(batch_correct.mean())

        if self.post_each_batch:
            state['metrics']['acc'] = self.avg.get()

    def on_epoch_end(self, state):
        state['metrics']['acc'] = self.avg.get()


class MetricsTable(tu.AutoStateDict):
    """
    Generate a HTML table with all the current metrics, to be displayed in
    Visdom.

    Args:
        post_each_batch (bool): whether to post on each batch or on epoch end.
            Default: True.
    """

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
        if self.post_each_batch and state.get('visdom_will_log', False):
            state['metrics']['table'] = self.make_html(state)

    def on_epoch_end(self, state):
        state['metrics']['table'] = self.make_html(state)


class Optimizer(tu.AutoStateDict):
    """
    Apply an optimizer's :code:`step()` and :code:`zero_grad()`.

    Args:
        opt (Optimizer): the optimizer to use
        accumulations (int): number of batches to accumulate gradients over
        clip_grad_norm (float or None): maximal norm of gradients to clip,
            before applying :code:`opt.step()`
        log_lr (bool): whether to log the current learning rates in the metrics
        log_mom (bool): whether to log the current momentum / beta1 in
            the metrics
    """

    def __init__(self,
                 opt,
                 accumulation=1,
                 clip_grad_norm=None,
                 log_lr=False,
                 log_mom=False):
        super(Optimizer, self).__init__()
        self.opt = opt
        self.accumulation = accumulation
        self.log_lr = log_lr
        self.log_mom = log_mom
        self.clip_grad_norm = clip_grad_norm

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
            if self.clip_grad_norm is not None:
                state['metrics']['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                    (p for pg in self.opt.param_groups for p in pg['params']),
                    self.clip_grad_norm)
            self.opt.step()


class LRSched(tu.AutoStateDict):
    """
    Call :code:`lr_sched.step()`.

    Args:
        sched (Scheduler): the scheduler to run
        metric (str or None): if :code:`step()` takes a value as an argument,
            that value should be in the state, and named here. Otherwise, just
            use None if it takes no argument.
        step_each_batch (bool): whether to call :code:`step()` on each batch or
            on each epoch.
    """

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
    """
    Move a value from the state to the metrics.

    Args:
        from_k (str): path in the state of the value to log (as accepted by
            :code:`torchelie.utils.dict_by_key()`
        to (str): metrics name
    """

    def __init__(self, from_k, to, post_each_batch=True):
        super(Log, self).__init__()
        self.from_k = from_k
        self.to = to
        self.post_each_batch = post_each_batch

    @torch.no_grad()
    def on_batch_end(self, state):
        if self.post_each_batch:
            state['metrics'][self.to] = dict_by_key(state, self.from_k)

    @torch.no_grad()
    def on_epoch_end(self, state):
        if not self.post_each_batch:
            state['metrics'][self.to] = dict_by_key(state, self.from_k)


class VisdomLogger(tu.AutoStateDict):
    """
    Log metrics to Visdom. It logs scalars and scalar tensors as plots, 3D and
    4D tensors as images, and strings as HTML.

    Args:
        visdom_env (str): name of the target visdom env
        log_every (int): batch logging freq. -1 logs on epoch ends only.
        prefix (str): prefix for all metrics name
    """

    def __init__(self, visdom_env='main', log_every=10, prefix=''):
        super(VisdomLogger, self).__init__(except_names=['vis'])
        self.vis = None
        self.log_every = log_every
        self.prefix = prefix
        if visdom_env is not None:
            self.vis = Visdom(env=visdom_env)
            self.vis.close()

    def on_batch_start(self, state):
        iters = state['iters']
        state['visdom_will_log'] = (self.log_every != -1
                                    and iters % self.log_every == 0)

    @torch.no_grad()
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
    """
    Log metrics to stdout. It logs scalars, scalar tensors and strings.

    Args:
        log_every (int): batch logging freq. -1 logs on epoch ends only.
        prefix (str): prefix for all metrics name
    """

    def __init__(self, log_every=10, prefix=''):
        super(StdoutLogger, self).__init__()
        self.log_every = log_every
        self.prefix = prefix

    def on_batch_end(self, state):
        iters = state['iters']
        if self.log_every != -1 and iters % self.log_every == 0:
            self.log(state['metrics'], state['epoch'], state['epoch_batch'])

    @torch.no_grad()
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


class ImageGradientVis:
    """
    Log gradients backpropagated to the input as a feature visualization mean.
    Works only for image data.
    """

    def on_batch_start(self, state):
        state['batch_gpu'][0].requires_grad = True

    @torch.no_grad()
    def on_batch_end(self, state):
        if not state.get('visdom_will_log', False):
            return

        x = state['batch_gpu'][0]
        grad_img = x.grad.abs().sum(1, keepdim=True)
        b, c, h, w = grad_img.shape
        gi_flat = grad_img.view(b, c, -1)
        cl = torch.kthvalue(gi_flat, int(grad_img[0].numel() * 0.99),
                            dim=-1)[0]
        grad_img = torch.min(grad_img, cl.unsqueeze(-1).unsqueeze(-1))
        m = gi_flat.min(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        M = gi_flat.max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        grad_img = (grad_img - m) / (M - m)
        img = (x + 1) / 2 * grad_img + 0.5 * (1 - grad_img)
        state['metrics']['feature_vis'] = img


class Checkpoint:
    """
    Save object to disk every so often.

    Args:
        filename_base (str): file will be saved as
            :code:`filename_base + '_' + number + '.pth'`.
        objects: what to save. It must have a :code:`state_dict()` member
    """

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
    """
    Polyak averaging (Exponential running average).

    Args:
        original (nn.Module): source module
        copy (nn.Module): averaged model
        beta (float): decay value
    """

    def __init__(self, original, copy, beta=0.999):
        self.original = original
        self.copy = copy
        self.beta = beta

    @torch.no_grad()
    def on_batch_end(self, state):
        for s, d in zip(self.original.parameters(), self.copy.parameters()):
            d.mul_(self.beta).add_(1 - self.beta, s)


class Counter(tu.AutoStateDict):
    """
    Count iterations and epochs. Mandatory as first callback as other callbacks
    may depend on it.

    It writes in the state the current epoch (as 'epoch'), the current
    iteration within batch ('epoch_batch') and the overall iteration number
    ('iters'):
    """

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
    """
    For image classification tasks, create a HTML report with best classified
    samples, worst classified samples, and samples closest to the boundary
    decision.

    Args:
        nb_show (int): how many samples to show
        classes (list of str): classes names
        post_each_batch (bool) whether to generate a new report on each batch
            or only on epoch end.
    """

    def __init__(self, nb_show, classes, post_each_batch=True):
        self.vis = CIVis(nb_show, classes, 1. / len(classes))
        self.post_each_batch = post_each_batch

    def on_epoch_start(self, state):
        if 'report' in state['metrics']:
            del state['metrics']['report']
        self.vis.reset()

    @torch.no_grad()
    def on_batch_end(self, state):
        pred, y, x = state['pred'], state['batch'][1], state['batch'][0]
        self.vis.analyze(x, pred, y)
        if self.post_each_batch and state.get('visdom_will_log', False):
            state['metrics']['report'] = self.vis.show()

    def on_epoch_end(self, state):
        state['metrics']['report'] = self.vis.show()


class ConfusionMatrix:
    """
    Generate a HTML confusion matrix.

    Args:
        labels (list of str): classes name
        normalize (bool): whether to show absolute or relative frequencies
    """

    def __init__(self, labels, normalize=False):
        self.labels = labels
        self.normalize = normalize

    def reset(self, state):
        state['cm'] = [[0] * len(self.labels) for _ in range(len(self.labels))]

    def on_epoch_start(self, state):
        self.reset(state)

    @torch.no_grad()
    def on_batch_end(self, state):
        import torchelie.utils as tu
        cm = state['cm']
        pred = tu.as_multiclass_shape(state['pred']).argmax(1)
        true = state['batch'][1]
        for p, t in zip(pred, true):
            cm[p][t] += 1

    def to_html(self, cm):
        s = "<tr><th></th><th>{}</th></tr>".format("</th><th>".join(
            self.labels))

        for l, row in zip(self.labels, cm):
            total = sum(row)
            if total == 0:
                row_str = "".join(["<td>0</td>" for _ in range(len(row))])
            else:
                row_str = ''.join([
                    ('<td style="background-color:rgb({0}, {0}, {0})">{1}'
                     '</td>').format(int(255 - x / total * 255), x)
                    for x in row
                ])
            s += "<tr><th>{}</th>{}</tr>".format(l, row_str)

        return "<table>" + s + "</table>"

    def on_epoch_end(self, state):
        state['metrics']['cm'] = self.to_html(state['cm'])


class CallRecipe(tu.AutoStateDict):
    """
    Call another recipe.

    Args:
        loop (Recipe): the recipe to call
        run_every (int): how often to call that recipe
        prefix (str): prefix of the metrics of the recipe
        init_fun (Callable or None): a fun to call before running the recipe
    """

    def __init__(self, loop, run_every=100, prefix='test', init_fun=None):
        super(CallRecipe, self).__init__(except_names=['init_fun', 'loop'])
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
