import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torchvision.datasets import MNIST
import torchvision.transforms as TF

import torchelie.nn as tnn
from torchelie.utils import kaiming, xavier
from torchelie.models import VggDebug, ResNetDebug



device = 'cuda'

ds = MNIST('.', download=True, transform=TF.Compose([TF.Resize(32),
    TF.ToTensor()]))
dl = torch.utils.data.DataLoader(ds, num_workers=4, batch_size=32)


#clf = VggDebug(in_ch=1).to(device)
clf = ResNetDebug(in_ch=1).to(device)

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
        print("Iter {}, loss {}, acc {}".format(
            iters, loss.item(), acc.item()))
    if iters == 2000:
        if acc > 0.93:
            print('PASS ({})'.format(acc))
            sys.exit(0)
        else:
            print('FAILURE ({})'.format(acc))
            sys.exit(1)
    iters += 1
