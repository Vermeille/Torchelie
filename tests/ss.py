import functools
import torchelie
import torchelie.recipes.imageclassifier

import torchelie.nn as tnn
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torchvision.transforms as TF
from torch.utils.data import DataLoader

tfms = torchelie.transforms.MultiBranch([
    TF.ToTensor(),
    TF.Compose([
        TF.RandomAffine(30, (0.1, 0.1), (0.9, 1.1)),
        TF.ColorJitter(0.2, 0.2, 0.2, 0.2),
        TF.RandomGrayscale(),
        TF.RandomHorizontalFlip(),
        TF.ToTensor()
    ]),
])

ds = CIFAR10('.', train=True, transform=tfms)
dl = DataLoader(ds,
                batch_size=64,
                shuffle=True,
                num_workers=4,
                pin_memory=True)

dst = CIFAR10('.', train=False, transform=TF.ToTensor())
dlt = DataLoader(dst,
                 batch_size=32,
                 shuffle=True,
                 num_workers=4,
                 pin_memory=True)


class Recipe(torch.nn.Module):
    def __init__(self, l2_radius=None, mixup='latent', lr=1e-3):
        super(Recipe, self).__init__()
        self.norm = torchelie.nn.ImageNetInputNorm()
        self.net = torch.nn.Sequential(
            torchelie.models.ResNetBone([
                '64:2', '64:1', '128:2', '128:1', '256:2', '256:1', '512:2',
                '512:1'
            ], functools.partial(tnn.Conv2dNormReLU, norm=None), tnn.ResBlock),
            torch.nn.AdaptiveAvgPool2d(1),
            torchelie.nn.Reshape(-1),
            torchelie.utils.kaiming(torch.nn.Linear(512, 512)),
        )
        self.estim = torch.nn.Linear(512, 10)
        self.l2_radius = l2_radius
        self.mixup = mixup
        self.sched = None
        self.lr = lr

    def l2_norm(self, x):
        if self.l2_radius is None:
            return x
        return F.normalize(x, dim=1) * self.l2_radius

    def forward(self, x):
        return self.estim(self.l2_norm(self.net(x)))

    def make_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch):
        x, y = batch
        pred = self.forward(self.norm(x))
        return {
            'pred': pred,
            'loss': torch.nn.functional.cross_entropy(pred, y)
        }

    def train_step(self, batch, opt):
        if self.sched is None:
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                    min_lr=1e-6, factor=0.5, patience=3000)
        ((x1, x2), y) = batch
        opt.zero_grad()
        xs = torch.cat([x1, x2], dim=0)
        fs = self.net(self.norm(xs))
        f1, f2 = fs[:fs.shape[0] // 2], fs[fs.shape[0] // 2:]

        if self.mixup == 'none':
            corr = torch.mm(self.l2_norm(f1), self.l2_norm(f2).t())
            ss_loss = torch.nn.functional.cross_entropy(
                corr,
                torch.arange(x1.shape[0]).long().to(x1.device))
        elif self.mixup == 'latent':
            fa, ya = torchelie.datasets.mixup(
                f2, f2[list(reversed(range(f2.shape[0])))],
                torch.arange(x1.shape[0]).long().to(x1.device),
                torch.arange(x1.shape[0] - 1, -1, -1).long().to(x1.device),
                f2.shape[0])

            corr = torch.mm(self.l2_norm(f1), self.l2_norm(fa).t())
            ss_loss = torchelie.loss.continuous_cross_entropy(corr, ya)
        ss_loss.backward()
        self.sched.step(ss_loss)

        pred = self.estim(f1.detach())
        clf_loss = torch.nn.functional.cross_entropy(pred, y)
        clf_loss.backward()
        opt.step()
        return {
            'loss': ss_loss,
            'ssloss': ss_loss.detach(),
            'clf_loss': clf_loss.detach(),
            'pred': pred,
            'lr': opt.param_groups[0]['lr']
        }


recipe = Recipe()
cook = torchelie.recipes.imageclassifier.ImageClassifier(
    recipe,
    train_callbacks=[
        torchelie.metrics.callbacks.WindowedMetricAvg('clf_loss'),
        torchelie.metrics.callbacks.WindowedMetricAvg('ssloss'),
        torchelie.metrics.callbacks.Log('batch.0.1', 'x_aug'),
        torchelie.metrics.callbacks.Log('batch.0.0', 'x'),
        torchelie.metrics.callbacks.Log('lr', 'lr'),
    ],
    visdom_env='selfsup',
    device='cuda')
cook(dl, dlt, 500)
