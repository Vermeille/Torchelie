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
import torchelie.models
from torchelie.models import VggClassCondGeneratorDebug, ProjPatch32
from torchelie.utils import nb_parameters, freeze, unfreeze
import torchelie.loss.gan.standard as gan_loss

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

tfms = TF.Compose([TF.Resize(32), TF.ToTensor()])
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
    G = Gen(in_noise=32, out_ch=1, num_classes=10).to(device)
    D = Discr(in_ch=1, out_ch=1, norm=None, num_classes=10).to(device)

    opt_G = Adam(G.parameters(), lr=2e-4, betas=(0., 0.999))
    opt_D = Adam(D.parameters(), lr=2e-4, betas=(0., 0.999))

    iters = 0
    for epoch in range(4):
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)

            z = torch.randn(32, 32).to(device)
            fake = G(z, y)

            opt_D.zero_grad()
            loss_fake = gan_loss.fake(D(fake.detach() * 2 - 1, y))
            loss_fake.backward()

            loss_true = gan_loss.real(D(x * 2 - 1, y))
            loss_true.backward()
            opt_D.step()

            opt_G.zero_grad()
            freeze(D)
            loss = gan_loss.generated(D(fake * 2 - 1, y))
            loss.backward()
            unfreeze(D)
            opt_G.step()

            if iters % 100 == 0:
                print("Iter {}, loss true {}, loss fake {}, loss G {}".format(
                    iters, loss_true.item(), loss_fake.item(), loss.item()))

            if iters % 10 == 0:
                vis.images(fake, opts=dict(store_history=True), win='fake')
            iters += 1

if opts.shapes_only:
    summary(VggClassCondGeneratorDebug)
    summary(Patch32)
else:
    train_net(VggClassCondGeneratorDebug, ProjPatch32)
