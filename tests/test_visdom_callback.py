import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import trackio
from torch.utils.data import DataLoader

import torchelie.callbacks as tcb
from torchelie.recipes import Recipe, TrainAndTest


@pytest.fixture
def visdom_client(monkeypatch):
    constructor = Mock()
    monkeypatch.setitem(sys.modules, 'visdom',
                        SimpleNamespace(Visdom=constructor))
    return constructor


def test_visdom_preserves_metric_types_and_existing_windows(visdom_client):
    logger = tcb.VisdomLogger('legacy', prefix='test_')
    image = torch.randn(2, 1, 4, 5, requires_grad=True)
    before = image.detach().clone()
    custom = Mock()
    logger.log(3, {
        'loss': 0.5,
        'tensor': torch.tensor(0.25, requires_grad=True),
        'report': '<b>Report</b>',
        'heatmap': torch.rand(4, 5),
        'image': torch.rand(3, 4, 5),
        'batch': image,
        'custom': custom,
    }, store_history=['test_image'])
    vis = visdom_client.return_value
    visdom_client.assert_called_once_with(env='legacy')
    vis.close.assert_not_called()
    assert vis.line.call_count == 2
    assert vis.line.call_args.kwargs['X'] == [3]
    assert vis.line.call_args.kwargs['Y'] == [0.25]
    vis.text.assert_called_once_with('<b>Report</b>', win='test_report',
                                     opts={'title': 'test_report'})
    vis.heatmap.assert_called_once()
    assert vis.image.call_args.kwargs['opts']['store_history']
    pixels = vis.images.call_args.args[0]
    assert isinstance(pixels, np.ndarray)
    assert pixels.shape == (2, 3, 4, 5)
    assert np.isfinite(pixels).all()
    assert torch.equal(image, before)
    custom.to_visdom.assert_called_once_with(vis, 'test_custom')


def test_disabled_visdom_needs_no_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, 'visdom', None)
    logger = tcb.VisdomLogger(None)
    state = {'iters': 0, 'metrics': {'unsupported': object()}}
    logger.on_batch_start(state)
    logger.on_batch_end(state)
    logger.on_epoch_end(state)
    assert not state['metrics_will_log']
    assert not state['visdom_will_log']


def test_missing_visdom_has_install_instructions(monkeypatch):
    monkeypatch.setitem(sys.modules, 'visdom', None)
    with pytest.raises(ImportError, match=r'Torchelie\[visdom\]'):
        tcb.VisdomLogger('legacy')


@pytest.mark.parametrize('frequency', [0, -2])
def test_invalid_visdom_frequency(visdom_client, frequency):
    with pytest.raises(ValueError, match='log_every'):
        tcb.VisdomLogger('legacy', log_every=frequency)
    visdom_client.assert_not_called()


@pytest.mark.parametrize('reverse', [False, True])
def test_trackio_and_visdom_log_together(visdom_client, monkeypatch, reverse):
    run = Mock(project='experiment')
    monkeypatch.setattr(trackio, 'init', Mock(return_value=run))
    loggers = [tcb.TrackioLogger('experiment', log_every=2, post_epoch_ends=False),
               tcb.VisdomLogger('legacy', log_every=3, post_epoch_ends=False)]
    loop = Recipe(lambda x: {'loss': float(x)}, range(5))
    loop.callbacks.add_prologue(tcb.Counter())
    loop.callbacks.add_callbacks([tcb.Log('loss', 'loss'), tcb.MetricsTable()])
    loop.callbacks.add_epilogues(loggers[::-1] if reverse else loggers)
    loop.run(1)
    assert [call.kwargs['step'] for call in run.log.call_args_list] == [0, 2, 4]
    vis = visdom_client.return_value
    assert [call.kwargs['X'] for call in vis.line.call_args_list] == [[0], [3]]
    assert vis.text.call_count == 2
    assert loop.callbacks.state['metrics_will_log']
    assert not loop.callbacks.state['visdom_will_log']


def test_visdom_epoch_only_logging(visdom_client):
    logger = tcb.VisdomLogger('legacy', log_every=-1)
    state = {'iters': 3, 'metrics': {'loss': 0.5}}
    logger.on_batch_start(state)
    logger.on_batch_end(state)
    assert not state['metrics_will_log']
    visdom_client.return_value.line.assert_not_called()
    logger.on_epoch_end(state)
    visdom_client.return_value.line.assert_called_once()


def test_recipes_still_default_to_trackio(visdom_client, monkeypatch):
    run = Mock(project='main')
    init = Mock(return_value=run)
    monkeypatch.setattr(trackio, 'init', init)
    loop = TrainAndTest(torch.nn.Linear(1, 1), lambda x: {}, lambda x: {},
                        DataLoader([0]), DataLoader([0]), checkpoint=None)
    loop.run(1)
    assert loop.trackio_run is run
    init.assert_called_once_with(project='main', embed=False)
    visdom_client.assert_not_called()
