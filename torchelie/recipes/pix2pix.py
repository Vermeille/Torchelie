from typing import Tuple
import copy
import os
import torch
import torchelie as tch
import torchelie.utils as tu
from torchelie.recipes.gan import GANRecipe
from torchelie.transforms import MultiBranch
import torchvision.transforms as TF
import torchelie.loss.gan.standard as gan_loss
from torchelie.loss.gan.penalty import zero_gp
from torchelie.datasets import UnlabeledImages, Pix2PixDataset
from torchelie.datasets import SideBySideImagePairsDataset
from torchelie.models import *
import torch.nn as nn


def get_dataset(dataset_specs: Tuple[str, str], img_size: int):
    ty, path = dataset_specs
    if ty == 'pix2pix':
        return Pix2PixDataset('~/.torch',
                              path,
                              split='train',
                              download=True,
                              transform=TF.Compose([
                                  TF.Resize(img_size),
                                  TF.RandomResizedCrop(img_size,
                                                       scale=(0.9, 1)),
                                  TF.RandomHorizontalFlip(),
                              ]))
    if ty == 'colorize':
        return UnlabeledImages(
            path,
            TF.Compose([
                TF.Resize(img_size),
                TF.RandomCrop(img_size),
                TF.RandomHorizontalFlip(),
                MultiBranch([
                    TF.Compose([
                        TF.Grayscale(3),
                        TF.ToTensor(),
                    ]),
                    TF.ToTensor(),
                ])
            ]))
    if ty == 'inpainting':
        return UnlabeledImages(
            path,
            TF.Compose([
                TF.Resize(img_size),
                TF.RandomCrop(img_size),
                TF.RandomHorizontalFlip(),
                TF.ToTensor(),
                MultiBranch([
                    TF.Compose([
                        TF.RandomErasing(p=1, value=(1., 1., 0)),
                    ]),
                    TF.Compose([])
                ])
            ]))
    if ty == 'edges':
        return UnlabeledImages(
            path,
            TF.Compose([
                TF.Resize(img_size),
                TF.CenterCrop(img_size),
                TF.RandomHorizontalFlip(),
                MultiBranch([
                    TF.Compose([
                        tch.transforms.Canny(),
                        TF.Grayscale(3),
                        TF.ToTensor(),
                    ]),
                    TF.Compose([
                        TF.ToTensor(),
                    ])
                ])
            ]))
    if ty == 'pairs':
        return SideBySideImagePairsDataset(
            path,
            TF.Compose([
                TF.Resize(img_size),
                TF.CenterCrop(img_size),
                TF.RandomHorizontalFlip(),
                TF.ToTensor(),
            ]))
    assert False, "dataset's type not understood"


@tu.experimental
def train(rank, world_size):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, type=lambda x: x.split(':'))
    parser.add_argument('--r0-gamma', type=float)
    parser.add_argument('--D-type', choices=['patch', 'unet'], default='patch')
    parser.add_argument('--l1-gain', default=0, type=float)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--from-ckpt')
    opts = parser.parse_args()

    G = pix2pix_256().to_instance_norm().to_equal_lr()
    G_polyak = copy.deepcopy(G)
    if opts.D_type == 'patch':
        D = residual_patch286()
        D.set_input_specs(6)
        D.to_equal_lr()
        r0_gamma = 0.1
    else:
        D = UNet([32, 64, 128, 256, 512], 1)
        D.set_decoder_num_layers(1)
        D.set_encoder_num_layers(1)
        D.set_input_specs(6)
        D.to_bilinear_sampling()
        D.leaky()
        tnn.utils.net_to_equal_lr(D, leak=0.2)
        r0_gamma = 0.00001
    r0_gamma = opts.r0_gamma or r0_gamma

    if rank == 0:
        print(G)
        print(D)

    G = torch.nn.parallel.DistributedDataParallel(G.to(rank), [rank], rank)
    D = torch.nn.parallel.DistributedDataParallel(D.to(rank), [rank], rank)

    ds = get_dataset(opts.dataset, 256)
    ds = torch.utils.data.DataLoader(ds,
                                     opts.batch_size,
                                     num_workers=4,
                                     shuffle=True,
                                     pin_memory=True,
                                     drop_last=True)

    def G_fun(batch) -> dict:
        x, y = batch
        G.train()
        D.train()
        out = G(x * 2 - 1)
        with D.no_sync():
            loss = gan_loss.generated(D(torch.cat([out, x], dim=1) * 2 - 1))
        loss += opts.l1_gain * F.l1_loss(out, y)
        loss.backward()
        return {'G_loss': loss.item()}

    class GradientPenalty:

        @tu.experimental
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

    gradient_penalty = GradientPenalty(r0_gamma)

    def D_fun(batch) -> dict:
        G.train()
        D.train()
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
            'fake_heatmap': torch.sigmoid(prob_fake.detach()),
            'prob_real': torch.sigmoid(prob_real).mean().item(),
            'real_loss': real_loss.item(),
            'g_norm': g_norm,
            'D-correct': (fake_correct + real_correct) / (2 * prob_fake.numel())
        }

    def test_fun(batch):
        x, _ = batch
        G_polyak.train()
        out = G_polyak(x * 2 - 1)
        return {'out': out.detach()}

    tag = f'pix2pix_{opts.dataset[0]}:{os.path.basename(opts.dataset[1])}'
    recipe = GANRecipe(G,
                       D,
                       G_fun,
                       D_fun,
                       test_fun,
                       ds,
                       checkpoint=tag if rank == 0 else None,
                       visdom_env=tag if rank == 0 else None)

    recipe.register('G_polyak', G_polyak)

    recipe.callbacks.add_callbacks([
        tch.callbacks.Optimizer(
            tch.optim.RAdamW(D.parameters(),
                             lr=2e-3,
                             betas=(0., 0.99),
                             weight_decay=0)),
        tch.callbacks.Log('out', 'out'),
        tch.callbacks.Log('batch.0', 'x'),
        tch.callbacks.Log('batch.1', 'y'),
        tch.callbacks.Log('fake_heatmap', 'fake_heatmap'),
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
        tch.callbacks.Polyak(G.module, G_polyak),
    ])
    recipe.test_loop.callbacks.add_callbacks([
        tch.callbacks.Log('out', 'polyak_out'),
    ])
    recipe.to(rank)
    if opts.from_ckpt is not None:
        recipe.load_state_dict(torch.load(opts.from_ckpt, map_location='cpu'))
    recipe.run(200)


if __name__ == '__main__':
    tu.parallel_run(train)
