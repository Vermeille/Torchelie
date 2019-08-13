import sys
import argparse

import crayons

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as TF

import torchelie.nn as tnn
import torchelie.models
from torchelie.models import VggDebug, ResNetDebug, PreactResNetDebug
from torchelie.utils import nb_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--dataset',
                    type=str,
                    choices=['mnist', 'cifar10'],
                    default='mnist')
parser.add_argument('--models', default='all')
parser.add_argument('--shapes-only', action='store_true')
opts = parser.parse_args()

device = 'cpu' if opts.cpu else 'cuda'

tfms = TF.Compose([TF.Resize(32), TF.ToTensor()])
if opts.dataset == 'mnist':
    ds = MNIST('.', download=True, transform=tfms)
if opts.dataset == 'cifar10':
    ds = CIFAR10('.', download=True, transform=tfms)
dl = torch.utils.data.DataLoader(ds,
                                 num_workers=4,
                                 batch_size=32,
                                 shuffle=True)
if opts.models == 'all':
    nets = [VggDebug, ResNetDebug, PreactResNetDebug]
else:
    nets = [torchelie.models.__dict__[m] for m in opts.models.split(',')]

def summary(Net):
    clf = Net(10, in_ch=1, debug=True).to(device)
    clf(torch.randn(32, 1, 32, 32).to(device))
    print('Nb parameters: {}'.format(nb_parameters(clf)))

def train_net(Net):
    clf = Net(10, in_ch=1).to(device)

    opt = Adam(clf.parameters())

    iters = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad()
        pred = clf(x)
        loss = F.cross_entropy(pred, y)
        loss.backward()
        opt.step()

        if iters % 100 == 0:
            acc = torch.mean((y == pred.argmax(dim=1)).float())
            print("Iter {}, loss {}, acc {}".format(iters, loss.item(),
                                                    acc.item()))
        if iters == 1500:
            if acc > 0.90:
                print(crayons.green('PASS ({})'.format(acc), bold=True))
            else:
                print(crayons.red('FAILURE ({})'.format(acc), bold=True))
            break
        iters += 1

for Net in nets:
    print(crayons.yellow('---------------------------------'))
    print(crayons.yellow('-- ' + Net.__name__))
    print(crayons.yellow('---------------------------------'))

    if opts.shapes_only:
        summary(Net)
    else:
        train_net(Net)
