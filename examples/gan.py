import argparse
import copy

import torch

from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CelebA, ImageFolder
import torchvision.transforms as TF

import torchelie.models as tmodels
from torchelie.models import VggGeneratorDebug, Patch32, VggDebug
from torchelie.utils import nb_parameters, freeze, unfreeze
from torchelie.optim import RAdamW
import torchelie.loss.gan.hinge as gan_loss
from torchelie.datasets import NoexceptDataset
from torchelie.transforms import AdaptPad
from torchelie.recipes.gan import GANRecipe
import torchelie.callbacks as tcb
from torch.optim import AdamW
from torchelie.recipes import Recipe

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--dataset',
                    type=str,
                    default='mnist')
parser.add_argument('--shapes-only', action='store_true')
opts = parser.parse_args()

device = 'cpu' if opts.cpu else 'cuda'
BS = 32

tfms = TF.Compose([
    TF.Resize(64),
    AdaptPad((64, 64)),
    TF.ToTensor()])
if opts.dataset == 'mnist':
    ds = MNIST('~/.cache/torch/mnist', download=True, transform=tfms)
elif opts.dataset == 'cifar10':
    ds = CIFAR10('~/.cache/torch/cifar10', download=True, transform=tfms)
elif opts.dataset == 'fashion':
    ds = FashionMNIST('~/.cache/torch/fashionmnist/',
                      download=True,
                      transform=tfms)
elif opts.dataset == 'celeba':
    tfms = TF.Compose([
        TF.CenterCrop(120),
        TF.Resize(64),
        AdaptPad((64, 64)),
        TF.ToTensor()])
    ds = CelebA('~/.cache/torch/celeba/', download=True, transform=tfms)
else:
    ds = NoexceptDataset(ImageFolder(opts.dataset, transform=tfms))
dl = torch.utils.data.DataLoader(ds,
                                 num_workers=4,
                                 batch_size=BS,
                                 shuffle=True)


def summary(Net):
    clf = Net(10, in_ch=1, debug=True).to(device)
    clf(torch.randn(32, 1, 32, 32).to(device))
    print('Nb parameters: {}'.format(nb_parameters(clf)))

def train_net(Gen, Discr):
    G = Gen(in_noise=128, out_ch=3)
    G_polyak = copy.deepcopy(G).eval()
    D = Discr(in_ch=3, out_ch=1)

    def G_fun(batch):
        z = torch.randn(BS, 128, device=device)
        fake = G(z)
        preds = D(fake * 2 - 1).squeeze()
        loss = gan_loss.generated(preds)
        loss.backward()
        return {'loss': loss.item(), 'imgs': fake.detach()}

    def G_polyak_fun(batch):
        print('POLYAK')
        z = torch.randn(BS, 128, device=device)
        fake = G_polyak(z)
        return {'imgs': fake.detach()}

    def D_fun(batch):
        z = torch.randn(BS, 128, device=device)
        fake = G(z)
        fake_loss = gan_loss.fake(D(fake * 2 - 1))
        fake_loss.backward()

        x = batch[0]
        x = x.expand(-1, 3, -1, -1)

        real_loss = gan_loss.real(D(x * 2 - 1))
        real_loss.backward()

        loss = real_loss.item() + fake_loss.item()
        return {'loss': loss, 'real_loss': real_loss.item(), 'fake_loss':
                fake_loss.item()}

    polyak_test = Recipe(G_polyak_fun, range(1))
    polyak_test.callbacks.add_callbacks([
        tcb.Log('imgs', 'imgs')
    ])
    loop = GANRecipe(G, D, G_fun, D_fun, dl, log_every=100).to(device)
    loop.G_loop.callbacks.add_callbacks([
        tcb.Optimizer(AdamW(G.parameters(), lr=1e-4, betas=(0., 0.99))),
    ])
    loop.register('G_polyak', G_polyak)
    loop.callbacks.add_callbacks([
        tcb.Polyak(G, G_polyak),
        tcb.CallRecipe(polyak_test, prefix='polyak'),
        tcb.Log('polyak_metrics.imgs', 'polyak_imgs'),
        tcb.Log('batch.0', 'x'),
        tcb.WindowedMetricAvg('real_loss'),
        tcb.WindowedMetricAvg('fake_loss'),
        tcb.Optimizer(AdamW(D.parameters(), lr=4e-4, betas=(0., 0.99))),
    ])
    loop.to(device).run(1000)


if opts.shapes_only:
    summary(VggGeneratorDebug)
    summary(Patch32)
else:
    train_net(tmodels.autogan_64, tmodels.snres_discr_4l)
