"""
Those are callbacks that can be attached to a recipe. They read a write from a
shared state.

WARNING: this might move to torchelie.recipes.callbacks
"""
import time
import copy
import os
from shutil import copyfile
from pathlib import Path
import torch
import torch.nn as nn
from visdom import Visdom

from torchelie.utils import dict_by_key, recursive_state_dict
from torchelie.callbacks.inspector import ClassificationInspector as CIVis
from torchelie.callbacks.inspector import SegmentationInspector as SIVis
import torchelie.utils as tu

from .avg import *

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TS = True
except Exception:
    pass


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
        val = state.get(self.name, None)
        if val is not None:
            self.avg.log(val)
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
        self.avg = RunningAvg()

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

    def __init__(self, post_each_batch=True, avg_type='window'):
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

    def __init__(self, post_each_batch=True, epoch_ends=True):
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
                html += '<tr><th>{}</th><td>{}</td></tr>'.format(k, round(v, 6))
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
            for group in self.opt.param_groups:
                for p in group['params']:
                    p.grad = None

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
            if self.accumulation != 1:
                for pg in self.opt.param_groups:
                    for p in pg['params']:
                        if p.grad is not None:
                            p.grad.data /= self.accumulation
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
        self.avg = RunningAvg()

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
        else:
            if self.metric is not None:
                self.avg.log(state['metrics'][self.metric])

    def on_epoch_end(self, state):
        if not self.step_each_batch:
            if self.metric is None:
                self.sched.step()
            else:
                self.sched.step(self.avg.get())
                self.avg = RunningAvg()


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
            try:
                state['metrics'][self.to] = dict_by_key(state, self.from_k)
            except:
                pass

    @torch.no_grad()
    def on_epoch_end(self, state):
        if not self.post_each_batch:
            try:
                state['metrics'][self.to] = dict_by_key(state, self.from_k)
            except:
                pass


class VisdomLogger:
    """
    Log metrics to Visdom. It logs scalars and scalar tensors as plots, 3D and
    4D tensors as images, and strings as HTML.

    Args:
        visdom_env (str): name of the target visdom env
        log_every (int): batch logging freq. -1 logs on epoch ends only.
        prefix (str): prefix for all metrics name
    """

    def __init__(self,
                 visdom_env='main',
                 log_every=10,
                 prefix='',
                 post_epoch_ends=True):
        self.vis = None
        self.log_every = log_every
        self.prefix = prefix
        self.post_epoch_ends = post_epoch_ends
        if visdom_env is not None:
            self.vis = Visdom(env=visdom_env)
            self.vis.close()

    def on_batch_start(self, state):
        iters = state['iters']
        state['visdom_will_log'] = (self.log_every != -1 and
                                    iters % self.log_every == 0)

    @torch.no_grad()
    def on_batch_end(self, state):
        iters = state['iters']
        if self.log_every != -1 and iters % self.log_every == 0:
            self.log(iters, state['metrics'])

    def on_epoch_end(self, state):
        if self.post_epoch_ends:
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
                                   opts=dict(title=name,
                                             store_history=name
                                             in store_history))
                elif x.dim() == 4:
                    m = x.min()
                    M = x.max()
                    if x.shape[1] == 1 or (m.item() < 0 or M.item() > 1):
                        x = x - m
                        x = x / (M - m + 1e-7)
                    if x.shape[1] == 1:
                        B, _, H, W = x.shape
                        x_flat = x.view(B * H, W)
                        import matplotlib.cm
                        import numpy as np
                        x_flat = matplotlib.cm.get_cmap('viridis')(x_flat)
                        x_flat = np.ascontiguousarray(x_flat[:, :, :3])
                        x_flat.shape = (B, H, W, 3)
                        x = x_flat.transpose(0, 3, 1, 2)
                    self.vis.images(x,
                                    win=name,
                                    opts=dict(title=name,
                                              store_history=name
                                              in store_history))
                else:
                    assert False, "incorrect tensor shape {} for {}".format(
                        repr(x.shape), name)
            else:
                assert False, "incorrect type {} for key {}".format(
                    x.__class__.__name__, name)


class TensorboardLogger:
    """
    Log metrics to Visdom. It logs scalars and scalar tensors as plots, 3D and
    4D tensors as images, and strings as HTML.

    Args:
        log_dir (str): path of the loging directory. Default to runs/CURRENT_DATETIME_HOSTNAME
        log_every (int): batch logging freq. -1 logs on epoch ends only.
        prefix (str): prefix for all metrics name
    """

    def __init__(self,
                 log_dir=None,
                 log_every=10,
                 prefix='',
                 post_epoch_ends=True):
        assert HAS_TS, ("Can't import Tensorboard. Some callbacks will not "
                        "work properly")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_every = log_every
        self.prefix = prefix
        self.post_epoch_ends = post_epoch_ends

    def on_batch_start(self, state):
        iters = state['iters']
        state['visdom_will_log'] = (self.log_every != -1 and
                                    iters % self.log_every == 0)

    @torch.no_grad()
    def on_batch_end(self, state):
        iters = state['iters']
        if self.log_every != -1 and iters % self.log_every == 0:
            self.log(iters, state['metrics'])

    def on_epoch_end(self, state):
        if self.post_epoch_ends:
            self.log(state['iters'], state['metrics'])

    def log(self, iters, xs, store_history=[]):
        for name, x in xs.items():
            name = self.prefix + name
            if isinstance(x, (float, int)):
                self.writer.add_scalar(name, x, iters)
            elif isinstance(x, str):
                self.writer.add_text(name, x, iters)
            elif isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    self.writer.add_scalar(name, x, iters)
                elif x.dim() == 2:
                    self.writer.add_image(name, x, iters, dataformats='HW')
                elif x.dim() == 3:
                    self.writer.add_image(name, x, iters, dataformats='CHW')
                elif x.dim() == 4:
                    self.writer.add_images(name, x, iters, dataformats='NCHW')
                else:
                    assert False, "incorrect tensor shape {} for {}".format(
                        repr(x.shape), name)
            else:
                assert False, "incorrect type {} for key {}".format(
                    x.__class__.__name__, name)


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
        img = self.compute_image(x).cpu()
        state['metrics']['feature_vis'] = img

    @torch.no_grad()
    def compute_image(self, x):
        grad_img = x.grad.abs().sum(1, keepdim=True)
        b, c, h, w = grad_img.shape
        gi_flat = grad_img.view(b, c, -1)
        cl = torch.kthvalue(gi_flat, int(grad_img[0].numel() * 0.99), dim=-1)[0]
        cl = cl.unsqueeze(-1).unsqueeze(-1)
        grad_img = torch.min(grad_img, cl) / cl
        x = x.detach()
        xm = x.min()
        xM = x.max()
        x = (x - xm) / (xM - xm)
        img = x * grad_img + 0.5 * (1 - grad_img)
        return img


class Checkpoint(tu.AutoStateDict):
    """
    Save object to disk every so often.

    Args:
        filename_base (str): a format string that is the filename. The format
            string can have keyword parameters that will be indexed in the
            state.
        objects: what to save. It must have a :code:`state_dict()` member
        max_saves (int): maximum number of checkpoints to save. Older
            checkpoints will be removed.
        key_best (func): key to determinte the best test. The value of the
            key parameter should be a function that takes a single argument
            and returns a key to use for sorting purposes.
    """

    def __init__(self, filename_base, objects, max_saves=10, key_best=None):
        super(Checkpoint, self).__init__(
            except_names=['objects', 'key_best', 'max_saves', 'key_best'])
        self.filename_base = filename_base
        self.objects = objects
        self.saved_fnames = []
        self.max_saves = max_saves
        self.best_save = float('-inf')
        self.key_best = key_best
        self.last_best_name = None

    def save(self, state):
        saved = recursive_state_dict(self.objects)
        nm = self.filename(state)
        try:
            Path(nm).parent.mkdir()
        except:
            pass
        torch.save(saved, nm)
        self.saved_fnames.append(nm)
        return saved

    def detach_save(self):
        nm = self.saved_fnames[-1]
        nm = nm.rsplit('.', 1)[0] + '_best.pth'
        copyfile(self.saved_fnames[-1], nm)
        try:
            os.remove(self.last_best_name)
        except Exception as e:
            print(str(e))
            pass
        self.last_best_name = nm

    def filename(self, state):
        return self.filename_base.format(**state)

    def on_epoch_end(self, state):
        saved = self.save(state)
        while len(self.saved_fnames) > self.max_saves:
            try:
                os.remove(self.saved_fnames[0])
            except Exception as e:
                print(str(e))
            self.saved_fnames = self.saved_fnames[1:]
        if self.key_best is not None and self.key_best(saved) >= self.best_save:
            self.best_save = self.key_best(saved)
            self.detach_save()


class Polyak:
    """
    Polyak averaging (Exponential moving average).

    Args:
        original (nn.Module): source module
        copy (nn.Module): averaged model
        beta (float): decay value
    """
    original: nn.Module
    copy: nn.Module
    beta: float

    @torch.no_grad()
    def __init__(self,
                 original: nn.Module,
                 copy: nn.Module,
                 beta: float = 0.999):
        self.original = original
        self.copy = copy
        self.beta = beta
        copy.load_state_dict(original.state_dict(), strict=False)

    @torch.no_grad()
    def on_batch_end(self, state):
        d = self.copy.state_dict()
        for name, value in self.original.state_dict().items():
            if value.dtype in [torch.float16, torch.float32, torch.float64]:
                d[name].mul_(self.beta).add_(value, alpha=1 - self.beta)
            else:
                d[name].copy_(value)


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
        paths = state['batch'][2] if len(state['batch']) > 2 else None
        self.vis.analyze(x, pred, y, paths=paths)
        if self.post_each_batch and state.get('visdom_will_log', False):
            state['metrics']['report'] = self.vis.show()

    def on_epoch_end(self, state):
        state['metrics']['report'] = self.vis.show()


class SegmentationInspector:
    """
    For image binnary segmentation tasks, create a HTML report with best segmented
    samples, worst segmented samples, and samples closest to the boundary
    decision.

    Args:
        nb_show (int): how many samples to show
        classes (list of str): classes names
        post_each_batch (bool) whether to generate a new report on each batch
            or only on epoch end.
    """

    def __init__(self, nb_show, classes, post_each_batch=True):
        self.vis = SIVis(nb_show, classes, 0.5)
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
        if 'f1' in state['metrics']:
            del state['metrics']['f1']

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
        s = "<tr><th>Pred\\True</th><th>{}</th></tr>".format("</th><th>".join(
            self.labels))

        for label, row in zip(self.labels, cm):
            total = sum(row)
            if total == 0:
                row_str = "".join(["<td>0</td>" for _ in range(len(row))])
            else:
                row_str = ''.join([
                    ('<td style="background-color:rgb({0}, {0}, {0})">{1}'
                     '</td>').format(int(255 - x / total * 255), x) for x in row
                ])
            s += "<tr><th>{}</th>{}</tr>".format(label, row_str)

        html = "<table>" + s + "</table>"
        return html

    def on_epoch_end(self, state):
        cm = state['cm']
        html = self.to_html(cm)
        if len(cm) == 2:
            precision = cm[0][0] / (cm[0][0] + cm[0][1] + 1e-8)
            recall = cm[0][0] / (cm[0][0] + cm[1][0] + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            state['metrics']['f1'] = f1
            html += '<p>F1: {:.2f}</p>'.format(f1)
        state['metrics']['cm'] = html


class CallRecipe:
    """
    Call another recipe.

    Args:
        loop (Recipe): the recipe to call
        run_every (int): how often to call that recipe
        prefix (str): prefix of the metrics of the recipe
        init_fun (Callable or None): a fun to call before running the recipe
    """

    def __init__(self, loop, run_every=100, prefix='test', init_fun=None):
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


class Throughput:
    """
    For debugging and benchmarking purposes: compute the througput (samples/s)
    of both the forward pass and the full iteration.

    It creates 4 new metrics:

    - :code:`iter_time`: number of seconds between two consecutives batch_start
        event.
    - :code:`iter_throughput` imgs/s for the whole iteration as
        :code:`batch_size / iter_time`
    - :code:`forward_time`: number of seconds between batch_start and batch_end
        events.
    - :code:`forward_throughput` imgs/s for the forward pass as
        :code:`batch_size / forward_time`
    """

    def __init__(self):
        self.b_start = None
        self.forward_avg = WindowAvg(10)
        self.it_avg = WindowAvg()

    def on_batch_start(self, state):
        now = time.time()
        if self.b_start:
            t = now - self.b_start
            self.it_avg.log(t)
            state['metrics']['iter_time'] = self.it_avg.get()
            state['metrics']['iter_throughput'] = len(
                state['batch'][0]) / self.it_avg.get()
        self.b_start = now

    def on_batch_end(self, state):
        t = time.time() - self.b_start
        self.forward_avg.log(t)
        state['metrics']['forward_time'] = self.forward_avg.get()
        state['metrics']['forward_throughput'] = len(
            state['batch'][0]) / self.forward_avg.get()
