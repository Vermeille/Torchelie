import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchelie.datasets import FastImageFolder
import torchvision.transforms as TF
import torchelie.transforms as TTF

import torchelie as tch
from torchelie.recipes.classification import CrossEntropyClassification


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--train-im-size', type=int, default=128)
    parser.add_argument('--test-im-size', type=int, default=192)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--test-every', type=int)
    parser.add_argument('--network', required=True)
    parser.add_argument('--train-path', required=True)
    parser.add_argument('--test-path', required=True)
    parser.add_argument('--optimizer',
                        choices=['sgd', 'adabelief'],
                        default='sgd')
    parser.add_argument('--weights')
    parser.add_argument('--pretrained')
    parser.add_argument('--from-ckpt')
    return parser.parse_args()


def build_transforms(train_im_size, test_im_size):
    tfm = TF.Compose([
        TF.RandomApply([TTF.PadToSquare()]),
        TTF.RandAugment(1, 30),
        TF.RandomResizedCrop(train_im_size),
        TF.RandomHorizontalFlip(),
        TF.ColorJitter(0.4, 0.4, 0.4, 0.05),
        TTF.Lighting(0.6),
        TF.ToTensor(),
        tch.nn.ImageNetInputNorm(),
    ])
    tfm_test = TF.Compose([
        #TTF.ResizedCrop(test_im_size),
        TF.Resize(256),
        TF.CenterCrop(224),
        TF.ToTensor(),
        tch.nn.ImageNetInputNorm(),
    ])

    return tfm, tfm_test


def get_datasets(train_path, test_path, train_im_size, test_im_size):
    train_tfms, tfms = build_transforms(train_im_size, test_im_size)
    ds = FastImageFolder(train_path, transform=train_tfms)
    dst = FastImageFolder(test_path, transform=tfms)
    return ds, dst


def get_model(network, num_classes, rank, weights=None, pretrained=None):
    model = tch.models.get_model(network,
                                 num_classes=num_classes,
                                 pretrained=pretrained)

    if weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))

    model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[rank],
                                                      output_device=rank)
    return model


def train(opts, rank, world_size):
    ds, dst = get_datasets(opts.train_path, opts.test_path, opts.train_im_size,
                           opts.test_im_size)

    model = get_model(opts.network, len(ds.classes), rank, opts.weights,
                      opts.pretrained)

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=4,
        batch_size=opts.batch_size,
        sampler=torch.utils.data.DistributedSampler(ds),
        persistent_workers=True,
        pin_memory=True,
        drop_last=True)

    dlt = torch.utils.data.DataLoader(dst,
                                      num_workers=4,
                                      batch_size=opts.batch_size,
                                      persistent_workers=True,
                                      prefetch_factor=4,
                                      pin_memory=True)

    env = ('imagenet_' + opts.network) if rank == 0 else None

    recipe = CrossEntropyClassification(model,
                                        dl,
                                        dlt,
                                        ds.classes,
                                        optimizer=opts.optimizer,
                                        lr=opts.lr,
                                        wd=opts.wd,
                                        beta1=0.9,
                                        log_every=len(dl) // 50,
                                        test_every=opts.test_every or len(dl),
                                        visdom_env=env,
                                        checkpoint=env,
                                        n_iters=len(dl) * opts.epochs)

    recipe.to(rank)

    if opts.from_ckpt is not None:
        recipe.load_state_dict(
            torch.load(opts.from_ckpt, map_location='cuda:' + str(rank)))

    if rank == 0:
        print(recipe)
    recipe.run(opts.epochs + 1)


if __name__ == '__main__':
    opts = get_args()
    tch.utils.parallel_run(train, opts)
