import sys
import argparse

import crayons

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchelie.optim import *

from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as TF

import torchelie.nn as tnn
import torchelie.models
from torchelie.models import VggDebug, ResNetDebug, PreactResNetDebug
from torchelie.utils import nb_parameters
from torchelie.recipes.classification import CrossEntropyClassification

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
    ds = MNIST('~/.cache/torch/mnist/', download=True, transform=tfms)
    dst = MNIST('~/.cache/torch/mnist/', download=True, transform=tfms, train=False)
if opts.dataset == 'cifar10':
    ds = CIFAR10('~/.cache/torch/cifar10', download=True, transform=tfms)
    dst = CIFAR10('~/.cache/torch/cifar10', download=True, transform=tfms, train=False)
dl = torch.utils.data.DataLoader(ds,
                                 num_workers=4,
                                 batch_size=32,
                                 shuffle=True)
dlt = torch.utils.data.DataLoader(dst,
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
    clf = Net(10, in_ch=1)


    clf_recipe = CrossEntropyClassification(clf,
                                            dl,
                                            dlt,
                                            ds.classes,
                                            device=device)

    acc = clf_recipe.fit(1)['test_metrics']['acc']

    if acc > 0.90:
        print(crayons.green('PASS ({})'.format(acc), bold=True))
    else:
        print(crayons.red('FAILURE ({})'.format(acc), bold=True))


for Net in nets:
    print(crayons.yellow('---------------------------------'))
    print(crayons.yellow('-- ' + Net.__name__))
    print(crayons.yellow('---------------------------------'))

    if opts.shapes_only:
        summary(Net)
    else:
        train_net(Net)
