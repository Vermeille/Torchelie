import os
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import trackio
from torch.utils.data import DataLoader

import torchelie.callbacks as tcb
from torchelie.recipes import GANRecipe, Recipe, TrainAndTest
from torchelie.recipes.recipebase import CallbacksRunner


@pytest.fixture
def init_run(monkeypatch):
    run = Mock(project='experiment')
    init = Mock(return_value=run)
    monkeypatch.setattr(trackio, 'init', init)
    return init


def test_scalars_html_and_custom_values(init_run, tmp_path):
    class CustomMetric:
        def to_trackio(self):
            return 12

    html_path = tmp_path / 'private.html'
    html_path.write_text('not a metric')
    logger = tcb.TrackioLogger('experiment', prefix='test_')
    logger.log(7, {
        'loss': torch.tensor(0.5, requires_grad=True),
        'count': np.int64(2),
        'report': '<b>Results</b>',
        'path': str(html_path),
        'custom': CustomMetric(),
    })
    metrics = logger.run.log.call_args.args[0]
    assert logger.run.log.call_args.kwargs == {'step': 7}
    assert metrics['test_loss'] == 0.5
    assert metrics['test_count'] == 2
    assert metrics['test_custom'] == 12
    assert '<b>Results</b>' in metrics['test_report']._html
    assert str(html_path) in metrics['test_path']._html
    assert 'not a metric' not in metrics['test_path']._html


@pytest.mark.parametrize('shape', [(4, 5), (1, 4, 5), (3, 4, 5),
                                  (4, 4, 5), (2, 1, 4, 5), (2, 3, 4, 5)])
def test_tensor_images(init_run, shape):
    logger = tcb.TrackioLogger('experiment')
    tensor = torch.randn(shape, requires_grad=True)
    before = tensor.detach().clone()
    logger.log(0, {'image': tensor})
    media = logger.run.log.call_args.args[0]['image']
    assert isinstance(media, trackio.Image)
    assert media._value.dtype == np.uint8
    assert media._value.ndim in (2, 3)
    assert torch.equal(tensor, before)


def test_image_intensity_and_constant_images(init_run):
    logger = tcb.TrackioLogger('experiment')
    logger.log(0, {'byte': torch.full((3, 2, 2), 128, dtype=torch.uint8),
                   'constant': torch.full((4, 5), -2.0)})
    metrics = logger.run.log.call_args.args[0]
    assert (metrics['byte']._value == 128).all()
    assert (metrics['constant']._value == 0).all()


@pytest.mark.parametrize('value,error', [(torch.ones(3), ValueError),
                                       (torch.ones(2, 4, 5), ValueError),
                                       (object(), TypeError)])
def test_unsupported_metrics_raise(init_run, value, error):
    logger = tcb.TrackioLogger('experiment')
    with pytest.raises(error):
        logger.log(0, {'invalid': value})
    logger.run.log.assert_not_called()


@pytest.mark.parametrize('frequency', [0, -2])
def test_invalid_frequency(init_run, frequency):
    with pytest.raises(ValueError, match='log_every'):
        tcb.TrackioLogger('experiment', log_every=frequency)
    init_run.assert_not_called()


def test_disabled_logging(init_run):
    logger = tcb.TrackioLogger(None)
    state = {'iters': 0, 'metrics': {'unsupported': object()}}
    logger.on_batch_start(state)
    logger.on_batch_end(state)
    logger.on_epoch_end(state)
    assert not state['metrics_will_log']
    assert logger.run is None
    init_run.assert_not_called()


def test_batch_and_epoch_schedule(init_run):
    logger = tcb.TrackioLogger('experiment', log_every=2,
                               post_epoch_ends=False)
    loop = Recipe(lambda value: {'loss': value}, [1, 2, 3, 4])
    loop.callbacks.add_prologue(tcb.Counter())
    loop.callbacks.add_callback(tcb.Log('loss', 'loss'))
    loop.callbacks.add_callback(tcb.MetricsTable())
    loop.callbacks.add_epilogue(logger)
    loop.run(1)
    assert [call.kwargs['step'] for call in logger.run.log.call_args_list] == [0, 2]
    assert not loop.callbacks.state['metrics_will_log']
    assert all('table' in call.args[0] for call in logger.run.log.call_args_list)

    logger.run.log.reset_mock()
    logger.log_every = -1
    logger.post_epoch_ends = True
    loop.run(1)
    logger.run.log.assert_called_once()
    assert logger.run.log.call_args.kwargs['step'] == 7
    logger.run.finish.assert_not_called()


def test_tensorboard_and_trackio_share_logging_schedule(init_run):
    runner = CallbacksRunner()
    runner.add_callback(tcb.TrackioLogger('experiment', log_every=2))
    runner.add_callback(tcb.TensorboardLogger(log_dir=None))
    runner.state['iters'] = 0
    runner('on_batch_start')
    assert runner.state['metrics_will_log']
    runner.state['iters'] = 1
    runner('on_batch_start')
    assert not runner.state['metrics_will_log']


def test_nested_recipes_share_run(init_run):
    loop = TrainAndTest(torch.nn.Linear(1, 1), lambda x: {'loss': 1.0},
                        lambda x: {'loss': 2.0}, DataLoader([0, 1]), DataLoader([0]),
                        trackio_project='experiment', test_every=1,
                        log_every=1, checkpoint=None)
    loop.callbacks.add_callback(tcb.Log('loss', 'loss'))
    loop.test_loop.callbacks.add_callback(tcb.Log('loss', 'loss'))
    loop.run(1)
    init_run.assert_called_once_with(project='experiment', embed=False)
    run = init_run.return_value
    assert loop.trackio_run is run
    assert any('loss' in call.args[0] for call in run.log.call_args_list)
    assert any('test_loss' in call.args[0] for call in run.log.call_args_list)
    run.finish.assert_not_called()


def test_gan_loggers_share_run(init_run, tmp_path):
    recipe = GANRecipe(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1),
                       lambda x: {}, lambda x: {}, lambda x: {}, [0],
                       trackio_project='experiment', checkpoint=str(tmp_path))
    loggers = [cb for loop in (recipe, recipe.G_loop, recipe.test_loop)
               for cb in loop.callbacks.callbacks()
               if isinstance(cb, tcb.TrackioLogger)]
    assert len(loggers) == 3
    assert recipe.trackio_run is init_run.return_value
    assert all(cb.run is init_run.return_value for cb in loggers)
    init_run.assert_called_once()


def test_mismatched_project_is_rejected(init_run):
    with pytest.raises(ValueError, match='run.project'):
        tcb.TrackioLogger('other-project', run=init_run.return_value)
    init_run.assert_not_called()


def test_local_trackio_persists_metrics_and_media(tmp_path):
    # A separate process isolates Trackio's import-time paths and global run.
    script = '''
from pathlib import Path
import torch
import trackio
from PIL import Image
from trackio.sqlite_storage import SQLiteStorage
from trackio.utils import MEDIA_DIR
from torchelie.callbacks import TrackioLogger

logger = TrackioLogger('integration')
test_logger = TrackioLogger('integration', prefix='test_', run=logger.run)
logger.log(3, {'loss': torch.tensor(0.25), 'report': '<b>Report</b>',
               'images': torch.ones(2, 3, 4, 5)})
test_logger.log(3, {'accuracy': 0.75})
logger.log(4, {'loss': 0.125})
trackio.finish()
records = SQLiteStorage.get_run_records('integration')
assert len(records) == 1, records
logs = SQLiteStorage.get_logs('integration', logger.run.name)
assert any(row.get('loss') == 0.25 and row['step'] == 3 for row in logs), logs
assert any(row.get('test_accuracy') == 0.75 for row in logs), logs
assert any(row.get('loss') == 0.125 and row['step'] == 4 for row in logs), logs
images = list(MEDIA_DIR.rglob('*.png'))
reports = list(MEDIA_DIR.rglob('*.html'))
assert images and reports
with Image.open(images[0]) as image:
    image.verify()
assert '<b>Report</b>' in reports[0].read_text()
'''
    env = dict(os.environ, TRACKIO_DIR=str(tmp_path / 'trackio'))
    for name in ('TRACKIO_SPACE_ID', 'TRACKIO_SERVER_URL', 'TRACKIO_BUCKET_ID',
                 'TRACKIO_DATASET_ID'):
        env.pop(name, None)
    result = subprocess.run([sys.executable, '-c', script], env=env,
                            cwd=tmp_path, capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
