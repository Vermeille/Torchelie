import torch
import torch.nn.functional as F
import torchelie.models as tmodels
from torchelie.metrics.avg import WindowAvg

import torchvision.transforms as TF
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torch.utils.data import DataLoader
from torchelie.optim import RAdamW
from torchelie.recipes import TrainAndCall
import torchelie.metrics as tcb


def train(model, loader):
    def train_step(batch):
        x = batch[0]
        x = x.expand(-1, 3, -1, -1)

        x2 = model(x * 2 - 1)
        loss = F.cross_entropy(x2, (x * 255).long())
        loss.backward()
        reconstruction = x2.argmax(dim=1).float() / 255.0
        return {'loss': loss, 'reconstruction': reconstruction}

    def after_train():
        imgs = model.sample(1, 4).expand(-1, 3, -1, -1)
        return {'imgs': imgs}

    opt = RAdamW(model.parameters(), lr=3e-3)
    trainer = TrainAndCall(model,
                           train_step,
                           after_train,
                           dl,
                           test_every=500,
                           visdom_env='pixelcnn')
    trainer.callbacks.add_callbacks([
        tcb.WindowedMetricAvg('loss'),
        tcb.Log('reconstruction', 'reconstruction'),
        tcb.Optimizer(opt, log_lr=True),
        tcb.LRSched(torch.optim.lr_scheduler.ReduceLROnPlateau(opt))
    ])
    trainer.test_loop.callbacks.add_callbacks([
        tcb.Log('imgs', 'imgs'),
    ])

    trainer.to('cuda')
    trainer.run(10)


tfms = TF.Compose([
    TF.Resize(32),
    TF.ToTensor(),
])

dl = DataLoader(FashionMNIST('~/.cache/torch/fashionmnist',
                             transform=tfms,
                             download=True),
                batch_size=32,
                shuffle=True,
                num_workers=4)

model = tmodels.PixelCNN(64, (32, 32), channels=3)
train(model, dl)
