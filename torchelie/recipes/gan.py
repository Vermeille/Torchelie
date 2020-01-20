import torch
import torchelie.utils as tu
import torchelie.callbacks as tcb
from torchelie.recipes.recipebase import Recipe


def GANRecipe(G,
             D,
             G_fun,
             D_fun,
             loader,
             *,
             visdom_env='main',
             checkpoint='model',
             log_every=10):
    def D_wrap(batch):
        tu.freeze(G)
        G.eval()
        tu.unfreeze(D)
        D.train()

        return D_fun(batch)

    def G_wrap(batch):
        tu.freeze(D)
        D.eval()
        tu.unfreeze(G)
        G.train()

        return G_fun(batch)

    D_loop = Recipe(D_wrap, loader)
    D_loop.register('G', G)
    D_loop.register('D', D)
    G_loop = Recipe(G_wrap, range(1))
    D_loop.G_loop = G_loop
    D_loop.register('G_loop', G_loop)

    def prepare_test(state):
        G_loop.callbacks.update_state({
            'epoch': state['epoch'],
            'iters': state['iters'],
            'epoch_batch': state['epoch_batch']
        })

    D_loop.callbacks.add_prologues(
        [tcb.Counter()])

    D_loop.callbacks.add_epilogues([
        tcb.CallRecipe(G_loop, 1, init_fun=prepare_test, prefix='G'),
        tcb.WindowedMetricAvg('loss'),
        tcb.Log('G_metrics.loss', 'G_loss'),
        tcb.Log('G_metrics.imgs', 'G_imgs'),
        tcb.VisdomLogger(visdom_env=visdom_env,
                         log_every=log_every),
        tcb.StdoutLogger(log_every=log_every),
        tcb.Checkpoint((visdom_env or 'model') + '/ckpt_{iters}.pth', D_loop)
    ])

    if checkpoint is not None:
        D_loop.callbacks.add_epilogues([
            tcb.Checkpoint(checkpoint + '/ckpt', D_loop)
        ])

    G_loop.callbacks.add_epilogues([
        tcb.Log('loss', 'loss'),
        tcb.Log('imgs', 'imgs'),
        tcb.WindowedMetricAvg('loss')
    ])

    return D_loop

