import torch
import torch.nn as nn
import torchelie.utils as tu
import torchelie.callbacks as tcb
from torchelie.recipes.recipebase import Recipe
from typing import Optional, Iterable, Any


@tu.experimental
def GANRecipe(G: nn.Module,
              D: nn.Module,
              G_fun,
              D_fun,
              test_fun,
              loader: Iterable[Any],
              *,
              visdom_env: Optional[str] = 'main',
              checkpoint: Optional[str] = 'model',
              test_every: int = 1000,
              log_every: int = 10,
              g_every: int = 1) -> Recipe:
    def D_wrap(batch):
        tu.freeze(G)
        tu.unfreeze(D)

        return D_fun(batch)

    def G_wrap(batch):
        tu.freeze(D)
        tu.unfreeze(G)

        return G_fun(batch)

    def test_wrap(batch):
        tu.freeze(G)
        tu.freeze(D)
        D.eval()
        G.eval()

        with torch.no_grad():
            out = test_fun(batch)

        D.train()
        G.train()
        return out

    class NoLim:
        def __init__(self):
            self.i = iter(loader)
            self.did_send = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.did_send:
                self.did_send = False
                raise StopIteration
            self.did_send = True
            try:
                return next(self.i)
            except:
                self.i = iter(loader)
                return next(self.i)

    D_loop = Recipe(D_wrap, loader)
    D_loop.register('G', G)
    D_loop.register('D', D)
    G_loop = Recipe(G_wrap, NoLim())
    D_loop.G_loop = G_loop
    D_loop.register('G_loop', G_loop)

    test_loop = Recipe(test_wrap, NoLim())
    D_loop.test_loop = test_loop
    D_loop.register('test_loop', test_loop)

    def G_test(state):
        G_loop.callbacks.update_state({
            'epoch': state['epoch'],
            'iters': state['iters'],
            'epoch_batch': state['epoch_batch']
        })

    def prepare_test(state):
        test_loop.callbacks.update_state({
            'epoch': state['epoch'],
            'iters': state['iters'],
            'epoch_batch': state['epoch_batch']
        })

    D_loop.callbacks.add_prologues([tcb.Counter()])

    D_loop.callbacks.add_epilogues([
        tcb.Log('imgs', 'G_imgs'),
        tcb.CallRecipe(G_loop, g_every, init_fun=G_test, prefix='G'),
        tcb.VisdomLogger(visdom_env=visdom_env, log_every=log_every),
        tcb.StdoutLogger(log_every=log_every),
        tcb.CallRecipe(test_loop,
                       test_every,
                       init_fun=prepare_test,
                       prefix='Test'),
    ])

    G_loop.callbacks.add_epilogues([
        tcb.WindowedMetricAvg('G_loss'),
        tcb.VisdomLogger(visdom_env=visdom_env,
                         log_every=log_every,
                         post_epoch_ends=False)
    ])

    if checkpoint is not None:
        test_loop.callbacks.add_epilogues([
            tcb.Checkpoint(checkpoint + '/ckpt_{iters}.pth', D_loop),
            tcb.VisdomLogger(visdom_env=visdom_env),
        ])

    return D_loop
