import torch
import torchelie as tch
import torchelie.utils as tu
from torchelie.recipes.gan import GANRecipe
from torchelie.transforms import MultiBranch
import torchvision.transforms as TF
import torchelie.loss.gan.standard as gan_loss
from torchelie.loss.gan.penalty import zero_gp
from torchelie.datasets.pix2pix import UnlabeledImages, Pix2PixDataset
from torchelie.models import residual_patch70, pix2pix_generator, pix2pix_res_dev
import torch.nn as nn

def train(rank, world_size):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--r0-gamma', default=0.0001, type=float)
    parser.add_argument('--from-ckpt')
    opts = parser.parse_args()

    G = pix2pix_res_dev()
    D = residual_patch70()
    D.set_input_specs(6)
    D.add_minibatch_stddev()
    D.to_equal_lr()

    for m in G.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            tu.kaiming(m, a=0., dynamic=True, mode='fan_out')
    #G.remove_batchnorm()

    if rank == 0:
        print(G)
        print(D)

    G  = torch.nn.parallel.DistributedDataParallel(G.to(rank), [rank], rank)
    D  = torch.nn.parallel.DistributedDataParallel(D.to(rank), [rank], rank)

    if True:
        ds = Pix2PixDataset(
            '~/.torch',
            opts.dataset,
            #split='train',
            download=True,
            transform=TF.Compose([
                TF.Resize(256),
                TF.RandomResizedCrop(256, scale=(0.9, 1)),
                TF.RandomHorizontalFlip(),
            ]))

    else:
        ds = UnlabeledImages(
                #'~/jaqen/face_clf_data/train/faces_ph/filtered/',
                '~/1bd/',
                TF.Compose([
                    TF.Resize(256),
                    TF.CenterCrop(256),
                    TF.RandomHorizontalFlip(),
                    MultiBranch([
                        TF.Compose([
                            TF.Grayscale(3),
                            TF.ToTensor(),
                        ]),
                        TF.ToTensor(),
                    ])
                    ])
                )

    sampler = None#torch.utils.data.WeightedRandomSampler([1] * len(ds), 10000*4, True)
    ds = torch.utils.data.DataLoader(ds, 2, num_workers=4, sampler=sampler,
            shuffle=sampler is None,
            pin_memory=True, drop_last=True)

    def G_fun(batch) -> dict:
        x, y = batch
        D.eval()
        out = G(x * 2 - 1)
        with D.no_sync():
            loss = gan_loss.generated(D(torch.cat([out, x], dim=1) * 2 - 1))
        loss.backward()
        return {'G_loss': loss.item()}

    class GradientPenalty:
        def __init__(self, gamma):
            self.gamma = gamma
            self.iters = 0
            self.last_norm = float('nan')

        def __call__(self, model, real, fake):
            if self.iters < 100 or self.iters % 4 == 0:
                real = real.detach()
                fake = fake.detach()
                gp, g_norm = zero_gp(model, real, fake)
                # Sync the gradient on the next backward
                if torch.any(torch.isnan(gp)):
                    gp.detach_()
                else:
                    (4 * self.gamma * gp).backward()
                self.last_norm = g_norm
            self.iters += 1
            return self.last_norm

    gradient_penalty = GradientPenalty(opts.r0_gamma)
    def D_fun(batch) -> dict:
        G.eval()
        x, y = batch
        with G.no_sync():
            with torch.no_grad():
                out = G(x * 2 - 1)
        fake = torch.cat([out, x], dim=1) * 2 - 1
        real = torch.cat([y, x], dim=1) * 2 - 1
        with D.no_sync():
            prob_fake = D(fake)
            fake_correct = prob_fake.detach().lt(0).int().eq(1).sum()
            fake_loss = gan_loss.fake(prob_fake)
            fake_loss.backward()

        with D.no_sync():
            g_norm = gradient_penalty(D, real, fake)

        prob_real = D(real)
        real_correct = prob_real.detach().gt(0).int().eq(1).sum()
        real_loss = gan_loss.real(prob_real)
        real_loss.backward()


        return {
            'out': out.detach(),
            'fake_loss': fake_loss.item(),
            'prob_fake': torch.sigmoid(prob_fake).mean().item(),
            'prob_real': torch.sigmoid(prob_real).mean().item(),
            'real_loss': real_loss.item(),
            'g_norm': g_norm,
            'D-correct': (fake_correct + real_correct) / (2 * prob_fake.numel())
        }

    def test_fun(_):
        return {}

    recipe = GANRecipe(G,
                       D,
                       G_fun,
                       D_fun,
                       test_fun,
                       ds,
                       checkpoint='face_inpaint' if rank == 0 else None,
                       visdom_env='main' if rank == 0 else None)

    recipe.callbacks.add_callbacks([
        tch.callbacks.Optimizer(
            tch.optim.RAdamW(D.parameters(),
                             lr=2e-3,
                             betas=(0., 0.99),
                             weight_decay=0)),
        tch.callbacks.Log('out', 'out'),
        tch.callbacks.Log('batch.0', 'x'),
        tch.callbacks.Log('batch.1', 'y'),
        #tch.callbacks.Log('batch.0.1', 'y'),
        tch.callbacks.WindowedMetricAvg('fake_loss', 'fake_loss'),
        tch.callbacks.WindowedMetricAvg('real_loss', 'real_loss'),
        tch.callbacks.WindowedMetricAvg('prob_fake', 'prob_fake'),
        tch.callbacks.WindowedMetricAvg('prob_real', 'prob_real'),
        tch.callbacks.WindowedMetricAvg('D-correct', 'D-correct'),
        tch.callbacks.Log('g_norm', 'g_norm')
    ])
    recipe.G_loop.callbacks.add_callbacks([
        tch.callbacks.Optimizer(
            tch.optim.RAdamW(G.parameters(),
                             lr=2e-3,
                             betas=(0., 0.99),
                             weight_decay=0)),
    ])
    recipe.to(rank)
    if opts.from_ckpt is not None:
        recipe.load_state_dict(torch.load(opts.from_ckpt, map_location='cpu'))
    recipe.run(200)

if __name__ == '__main__':
    tu.parallel_run(train)
