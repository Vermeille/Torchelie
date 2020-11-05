import torch
import torchelie.utils as tu
import torchelie.callbacks as tcb
from torchelie.recipes.recipebase import Recipe


def GANRecipe(G,
              D,
              G_fun,
              D_fun,
              test_fun,
              loader,
              *,
              visdom_env='main',
              checkpoint='model',
              test_every=1000,
              log_every=10,
              g_every=1):
    """
    Runs the main function.

    Args:
        G: (todo): write your description
        D: (todo): write your description
        G_fun: (todo): write your description
        D_fun: (callable): write your description
        test_fun: (todo): write your description
        loader: (todo): write your description
        visdom_env: (todo): write your description
        checkpoint: (bool): write your description
        test_every: (bool): write your description
        log_every: (todo): write your description
        g_every: (todo): write your description
    """
    def D_wrap(batch):
        """
        Wrap a single array

        Args:
            batch: (todo): write your description
        """
        tu.freeze(G)
        tu.unfreeze(D)

        return D_fun(batch)

    def G_wrap(batch):
        """
        Wrap a g ( g g ).

        Args:
            batch: (todo): write your description
        """
        tu.freeze(D)
        tu.unfreeze(G)

        return G_fun(batch)

    def test_wrap(batch):
        """
        Decorator into train test.

        Args:
            batch: (todo): write your description
        """
        tu.freeze(G)
        tu.freeze(D)
        D.eval()
        G.eval()

        out = test_fun(batch)

        D.train()
        G.train()
        return out

    D_loop = Recipe(D_wrap, loader)
    D_loop.register('G', G)
    D_loop.register('D', D)
    G_loop = Recipe(G_wrap, range(1))
    D_loop.G_loop = G_loop
    D_loop.register('G_loop', G_loop)

    test_loop = Recipe(test_wrap, range(1))
    D_loop.test_loop = test_loop
    D_loop.register('test_loop', test_loop)

    def G_test(state):
        """
        Test if g_test was run

        Args:
            state: (todo): write your description
        """
        G_loop.callbacks.update_state({
            'epoch': state['epoch'],
            'iters': state['iters'],
            'epoch_batch': state['epoch_batch']
        })

    def prepare_test(state):
        """
        Prepare a test state.

        Args:
            state: (todo): write your description
        """
        test_loop.callbacks.update_state({
            'epoch': state['epoch'],
            'iters': state['iters'],
            'epoch_batch': state['epoch_batch']
        })

    D_loop.callbacks.add_prologues([tcb.Counter()])

    D_loop.callbacks.add_epilogues([
        tcb.CallRecipe(G_loop, g_every, init_fun=G_test, prefix='G'),
        tcb.WindowedMetricAvg('loss'),
        tcb.Log('G_metrics.loss', 'G_loss'),
        tcb.Log('imgs', 'G_imgs'),
        tcb.VisdomLogger(visdom_env=visdom_env, log_every=log_every),
        tcb.StdoutLogger(log_every=log_every),
        tcb.CallRecipe(test_loop,
                       test_every,
                       init_fun=prepare_test,
                       prefix='Test'),
    ])

    G_loop.callbacks.add_epilogues(
        [tcb.Log('loss', 'loss'),
         tcb.WindowedMetricAvg('loss')])

    if checkpoint is not None:
        test_loop.callbacks.add_epilogues(
            [tcb.Checkpoint(checkpoint + '/ckpt_{iters}.pth', D_loop)])

    return D_loop


if __name__ == '__main__':
    from torchelie.models import autogan_128, snres_discr
    import torchelie.loss.gan.hinge as gan_loss
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as TF
    from torchelie.optim import RAdamW
    import torchelie as tch
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--noise-size', type=int, default=128)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--img-dir', required=True)
    opts = parser.parse_args()

    device = opts.device
    G = tch.models.VggGeneratorDebug(opts.noise_size, out_sz=opts.img_size)
    D = snres_discr(1, input_sz=opts.img_size, max_channels=512).to(device)

    optG = RAdamW(G.parameters(), 2e-4, betas=(0., 0.99), weight_decay=0)
    optD = RAdamW(D.parameters(), 2e-4, betas=(0., 0.99), weight_decay=0)

    def G_train(batch):
        """
        Train g_train model

        Args:
            batch: (todo): write your description
        """
        tu.fast_zero_grad(G)
        imgs = G(torch.randn(opts.batch_size, opts.noise_size, device=device))
        score = gan_loss.generated(D(imgs * 2 - 1))
        score.backward()
        optG.step()
        return {'loss': score.item()}

    def D_train(batch):
        """
        Perform training step.

        Args:
            batch: (todo): write your description
        """
        tu.fast_zero_grad(D)
        fake = G(torch.randn(opts.batch_size, opts.noise_size, device=device))
        fake_loss = gan_loss.fake(D(fake * 2 - 1))
        fake_loss.backward()
        optD.step()

        tu.fast_zero_grad(D)
        real_loss = gan_loss.real(D(batch[0] * 2 - 1))
        real_loss.backward()
        optD.step()
        return {
            'imgs': fake.detach(),
            'loss': real_loss.item() + fake_loss.item()
        }

    def test(batch):
        """
        Decor function and return a dictionary.

        Args:
            batch: (int): write your description
        """
        return {}

    tfm = TF.Compose([
        TF.Resize(opts.img_size),
        TF.CenterCrop(opts.img_size),
        TF.ToTensor()
    ])
    dl = torch.utils.data.DataLoader(ImageFolder(opts.img_dir,
                                                 transform=tfm),
                                     num_workers=4,
                                     shuffle=True,
                                     pin_memory=True,
                                     batch_size=opts.batch_size)

    recipe = GANRecipe(G, D, G_train, D_train, test, dl, visdom_env='gan')
    recipe.callbacks.add_callbacks([
        #tch.callbacks.Optimizer(optG)
        tcb.Log('batch.0', 'x'),
    ])
    recipe.to(device)
    recipe.run(500)
