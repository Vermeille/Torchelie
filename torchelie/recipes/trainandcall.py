import copy

import torch

import torchelie.metrics.callbacks as tcb

from .trainandtest import TrainAndTest


def TrainAndCall(model,
                 train_fun,
                 test_fun,
                 train_loader,
                 test_every=100,
                 visdom_env='main',
                 log_every=10,
                 device='cpu'):
    """
    Training loop, calls `model.after_train()` after every `test_every`
    iterations. Displays Loss.

    It logs the same things as TrainAndCallBase, plus whatever is returned by
    `model.after_train()`

    Args:
        model (nn.Model): a model
            The model must define:
                - `model.make_optimizer()`
                - `model.after_train()` returns a dict
                - `model.train_step(batch, opt)` returns a dict with key loss
        train_loader (DataLoader): Training set dataloader
        visdom_env (str): name of the visdom environment to use, or None for
            not using Visdom (default: None)
        test_every (int): testing frequency, in number of iterations (default:
            1000)
        train_callbacks (list of Callback): additional training callbacks
            (default: [])
        test_callbacks (list of Callback): additional testing callbacks
            (default: [])
        device: a torch device (default: 'cpu')
    """

    def test_fun_wrap(_):
        return test_fun()

    return TrainAndTest(model,
                        train_fun,
                        test_fun_wrap,
                        train_loader=train_loader,
                        test_loader=range(1),
                        test_every=test_every,
                        visdom_env=visdom_env,
                        log_every=log_every,
                        device=device)
