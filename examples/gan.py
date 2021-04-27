import argparse
import copy

import torch

from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as TF

import torchelie as tch
import torchelie.loss.gan.hinge as gan_loss
from torchelie.recipes.gan import GANRecipe
import torchelie.callbacks as tcb
from torchelie.recipes import Recipe

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
opts = parser.parse_args()

device = 'cpu' if opts.cpu else 'cuda'
BS = 32

tfms = TF.Compose([
    TF.Resize(64),
    tch.transforms.AdaptPad((64, 64)),
    TF.RandomHorizontalFlip(),
    TF.ToTensor()])
ds = CIFAR10('~/.cache/torch/cifar10', download=True, transform=tfms)
dl = torch.utils.data.DataLoader(ds,
                                 num_workers=4,
                                 batch_size=BS,
                                 shuffle=True)


def train_net(Gen, Discr):
    G = Gen(in_noise=128, out_ch=3)
    G_polyak = copy.deepcopy(G).eval()
    D = Discr()
    print(G)
    print(D)

    def G_fun(batch):
        z = torch.randn(BS, 128, device=device)
        fake = G(z)
        preds = D(fake * 2 - 1).squeeze()
        loss = gan_loss.generated(preds)
        loss.backward()
        return {'loss': loss.item(), 'imgs': fake.detach()}

    def G_polyak_fun(batch):
        z = torch.randn(BS, 128, device=device)
        fake = G_polyak(z)
        return {'imgs': fake.detach()}

    def D_fun(batch):
        z = torch.randn(BS, 128, device=device)
        fake = G(z)
        fake_loss = gan_loss.fake(D(fake * 2 - 1))
        fake_loss.backward()

        x = batch[0]

        real_loss = gan_loss.real(D(x * 2 - 1))
        real_loss.backward()

        loss = real_loss.item() + fake_loss.item()
        return {'loss': loss, 'real_loss': real_loss.item(), 'fake_loss':
                fake_loss.item()}

    loop = GANRecipe(G, D, G_fun, D_fun, G_polyak_fun, dl, log_every=100).to(device)
    loop.register('polyak', G_polyak)
    loop.G_loop.callbacks.add_callbacks([
        tcb.Optimizer(tch.optim.RAdamW(G.parameters(), lr=1e-4, betas=(0., 0.99))),
        tcb.Polyak(G, G_polyak),
    ])
    loop.register('G_polyak', G_polyak)
    loop.callbacks.add_callbacks([
        tcb.Log('batch.0', 'x'),
        tcb.WindowedMetricAvg('real_loss'),
        tcb.WindowedMetricAvg('fake_loss'),
        tcb.Optimizer(tch.optim.RAdamW(D.parameters(), lr=4e-4, betas=(0., 0.99))),
    ])
    loop.test_loop.callbacks.add_callbacks([
        tcb.Log('imgs', 'polyak_imgs'),
        tcb.VisdomLogger('main', prefix='test')
    ])
    loop.to(device).run(100)


train_net(tch.models.autogan_64, tch.models.snres_discr_4l)
