import sys
import argparse

import crayons

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as TF

import torchelie.nn as tnn
import torchelie.models
import torchelie as tch
from torchelie.models import ClassCondResNetDebug
from torchelie.utils import nb_parameters
from torchelie.recipes.classification import Classification
from torchelie.optim import RAdamW

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--dataset',
                    type=str,
                    choices=['mnist', 'cifar10'],
                    default='mnist')
parser.add_argument('--models', default='all')
opts = parser.parse_args()

device = 'cpu' if opts.cpu else 'cuda'


class TrueOrFakeLabelDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.classes = ['Fake', 'True']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if torch.randn(1).item() < 0:
            return x, 1, y
        return x, 0, torch.randint(0, 10, (1, )).item()


tfms = TF.Compose([TF.Resize(32), TF.ToTensor()])
ds = TrueOrFakeLabelDataset(
    MNIST('~/.cache/torch/mnist', download=True, transform=tfms))
dt = TrueOrFakeLabelDataset(
    MNIST('~/.cache/torch/mnist', download=True, transform=tfms, train=False))
dl = torch.utils.data.DataLoader(ds,
                                 num_workers=4,
                                 batch_size=32,
                                 shuffle=True)
dlt = torch.utils.data.DataLoader(dt,
                                  num_workers=4,
                                  batch_size=32,
                                  shuffle=True)


def train_net():
    model = ClassCondResNetDebug(2, 10, in_ch=1)

    def train_step(batch):
        x, y, z = batch

        out = model(x, z)
        loss = F.cross_entropy(out, y)
        loss.backward()

        return {'loss': loss, 'pred': out}

    def validation_step(batch):
        x, y, z = batch

        out = model(x, z)
        loss = F.cross_entropy(out, y)
        return {'loss': loss, 'pred': out}

    clf = Classification(model, train_step, validation_step, dl, dlt,
                         ds.classes).to(device)
    clf.callbacks.add_callbacks(
        [tch.callbacks.Optimizer(tch.optim.RAdamW(model.parameters()))])
    clf.run(2)


train_net()
