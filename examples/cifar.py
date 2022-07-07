"""
This example demonstrates how to learn CIFAR-10 with Torchelie. It can be
trivially modified to fit another dataset or model.

Better than that, make sure to check the Classification Recipe's builtin
command line interface that allows to fit a model to an image dataset without
writing a single line of code. It is good to quickly estimate how hard a
dataset is to learn by fitting a default model with default transforms and
hyperparameters.
"""
import argparse
from contextlib import suppress

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import CIFAR10
import torchvision.transforms as TF

import torchelie as tch
from torchelie.recipes.classification import CrossEntropyClassification


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--network', required=True)
    return parser.parse_args()


def build_transforms():
    tfms = TF.Compose([
        TF.Resize(32),
        TF.ToTensor(),
        TF.Normalize([0.5] * 3, [0.5] * 3, True),
    ])
    train_tfms = TF.Compose([
        TF.RandomCrop(32, padding=4),
        TF.RandomHorizontalFlip(),
        TF.ToTensor(),
        TF.Normalize([0.5] * 3, [0.5] * 3, True),
    ])
    return tfms, train_tfms


def get_datasets():
    tfms, train_tfms = build_transforms()
    ds = CIFAR10('~/.cache/torch/cifar10', download=True, transform=train_tfms)
    dst = CIFAR10('~/.cache/torch/cifar10',
                  transform=tfms,
                  download=True,
                  train=False)
    return ds, dst


def train():
    opts = get_args()
    ds, dst = get_datasets()
    m = opts.network
    model = tch.models.get_model(m, num_classes=10)

    with suppress(AttributeError):
        # Use a stem suitable for cifar10
        model.features.input.set_input_specs(32)

    dl = torch.utils.data.DataLoader(ds,
                                     num_workers=4,
                                     batch_size=128,
                                     pin_memory=True,
                                     persistent_workers=True,
                                     shuffle=True)
    dlt = torch.utils.data.DataLoader(dst,
                                      num_workers=4,
                                      batch_size=256,
                                      persistent_workers=True,
                                      pin_memory=True)
    recipe = CrossEntropyClassification(model,
                                        dl,
                                        dlt,
                                        ds.classes,
                                        optimizer='adabelief',
                                        lr=opts.lr,
                                        wd=opts.wd,
                                        beta1=0.9,
                                        log_every=100,
                                        test_every=len(dl) * opts.epochs // 10,
                                        visdom_env='cifar_' + m,
                                        n_iters=len(dl) * opts.epochs)

    recipe.to(opts.device)
    print(recipe)
    recipe.run(opts.epochs + 1)


if __name__ == '__main__':
    train()
