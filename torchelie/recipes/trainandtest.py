import torch

import torchelie.callbacks as tcb
from torchelie.recipes import Recipe


def TrainAndTest(model,
                 train_fun,
                 test_fun,
                 train_loader,
                 test_loader,
                 *,
                 test_every=100,
                 visdom_env='main',
                 log_every=10,
                 checkpoint='model',
                 key_best=None):
    """
    Two nested loops, usually one for training and one for testing, but can
    serve other purposes. The model is automatically registered and
    checkpointed as :code:`checkpoint['model']`, and put in eval mode when
    testing.

    Training callbacks:

    - Counter for counting iterations, connected to the testing loop as well
    - VisdomLogger
    - StdoutLogger

    Testing:

    Testing loop is in :code:`.test_loop`.

    Testing callbacks:

    - VisdomLogger
    - StdoutLogger
    - Checkpoint

    Args:
        model (nn.Model): a model
        train_fun (Callabble): a function that takes a batch as a single
            argument, performs a training step and return a dict of values to
            populate the recipe's state.
        test_fun (Callable): a function taking a batch as a single argument
            then performs something to evaluate your model and returns a dict
            to populate the state.
        train_loader (DataLoader): Training set dataloader
        test_loader (DataLoader): Testing set dataloader
        test_every (int): testing frequency, in number of iterations (default:
            100)
        visdom_env (str): name of the visdom environment to use, or None for
            not using Visdom (default: None)
        log_every (int): logging frequency, in number of iterations (default:
            100)
        checkpoint (str): checkpointing path or None for no checkpointing
        key_best (function or None): a key function for comparing states.
            Checkpointing the greatest.
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
            tcb.Checkpoint(checkpoint + '/ckpt_{iters}.pth',
                           train_loop,
                           key_best=key_best)
        ])

    return train_loop
