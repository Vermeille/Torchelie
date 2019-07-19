import sys
import argparse

import crayons

from visdom import Visdom

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as TF

import torchelie.nn as tnn
from torchelie.utils import kaiming, xavier
from torchelie.models import VggImg2ImgGeneratorDebug, Patch32
from torchelie.utils import nb_parameters, freeze, unfreeze
import torchelie.loss.gan.standard as gan_loss
import torchelie.transforms as TTF

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--dataset',
                    type=str,
                    choices=['mnist', 'cifar10'],
                    default='mnist')
parser.add_argument('--shapes-only', action='store_true')
opts = parser.parse_args()

device = 'cpu' if opts.cpu else 'cuda'

vis = Visdom(env='gan_dbg')

tfms = TF.Compose([
    TF.Resize(32),
    TTF.MultiBranch([TF.ToTensor(),
                     TF.Compose([TTF.Canny(), TF.ToTensor()])])
])

if opts.dataset == 'mnist':
    ds = MNIST('.', download=True, transform=tfms)
if opts.dataset == 'cifar10':
    ds = CIFAR10('.', download=True, transform=tfms)
dl = torch.utils.data.DataLoader(ds,
                                 num_workers=4,
                                 batch_size=32,
                                 shuffle=True)


def summary(Net):
    clf = Net(10, in_ch=1, debug=True).to(device)
    clf(torch.randn(32, 1, 32, 32).to(device))
    print('Nb parameters: {}'.format(nb_parameters(clf)))


def train_net(Gen, Discr):
    G = Gen(in_noise=32, out_ch=3).to(device)
    D = Discr(in_ch=4, out_ch=1, norm=None).to(device)

    opt_G = Adam(G.parameters(), lr=2e-4, betas=(0., 0.999))
    opt_D = Adam(D.parameters(), lr=2e-4, betas=(0., 0.999))

    iters = 0
    for epoch in range(10):
        for (x, x2), y in dl:
            x = x.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            z = torch.randn(y.size(0), 32).to(device)
            fake = G(z, x2)

            opt_D.zero_grad()
            loss_fake = gan_loss.fake(D(torch.cat([fake.detach(), x2], dim=1) * 2 - 1))
            loss_fake.backward()

            loss_true = gan_loss.real(D(torch.cat([x, x2], dim=1) * 2 - 1))
            loss_true.backward()
            opt_D.step()

            opt_G.zero_grad()
            freeze(D)
            loss = gan_loss.generated(D(torch.cat([fake, x2], dim=1) * 2 - 1))
            loss.backward()
            unfreeze(D)
            opt_G.step()

            if iters % 100 == 0:
                print("Iter {}, loss true {}, loss fake {}, loss G {}".format(
                    iters, loss_true.item(), loss_fake.item(), loss.item()))

            if iters % 10 == 0:
                vis.images(x2, win='canny')
                vis.images(fake, opts=dict(store_history=True), win='fake')
            iters += 1


if opts.shapes_only:
    summary(VggImg2ImgGeneratorDebug)
    summary(Patch32)
else:
    train_net(VggImg2ImgGeneratorDebug, Patch32)
