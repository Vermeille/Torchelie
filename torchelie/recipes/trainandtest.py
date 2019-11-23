import copy

import torch

import torchelie.metrics.callbacks as tcb
from torchelie.recipes.recipebase import Recipe


def TrainAndTest(model,
                 train_fun,
                 test_fun,
                 train_loader,
                 test_loader,
                 test_every=100,
                 visdom_env='main',
                 log_every=10,
                 checkpoint='model'):
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
    """

    def eval_call(batch):
        model.eval()
        with torch.no_grad():
            out = test_fun(batch)
        model.train()
        return out

    train_loop = Recipe(train_fun, train_loader)
    train_loop.register('model', model)

    test_loop = Recipe(eval_call, test_loader)
    train_loop.test_loop = test_loop
    train_loop.register('test_loop', test_loop)

    def prepare_test(state):
        test_loop.callbacks.update_state({
            'epoch': state['epoch'],
            'iters': state['iters'],
            'epoch_batch': state['epoch_batch']
        })

    train_loop.callbacks.add_prologues([tcb.Counter()])
    train_loop.callbacks.add_epilogues([
        tcb.CallRecipe(test_loop, test_every, init_fun=prepare_test),
        tcb.VisdomLogger(visdom_env=visdom_env, log_every=log_every),
        tcb.StdoutLogger(log_every=log_every),
    ])

    test_loop.callbacks.add_epilogues([
        tcb.VisdomLogger(visdom_env=visdom_env, log_every=-1, prefix='test_'),
        tcb.StdoutLogger(log_every=-1, prefix='Test'),
    ])

    if checkpoint is not None:
        test_loop.callbacks.add_epilogues([
            tcb.Checkpoint(checkpoint + '/ckpt', train_loop)
        ])

    return train_loop
