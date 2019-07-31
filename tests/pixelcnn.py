import torch
import torch.nn.functional as F
import torchelie.models as tmodels

import torchvision.transforms as TF
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
from torch.optim import Adam

from visdom import Visdom

device = 'cpu'

vis = Visdom(env='pixcnn')

pixcnn = tmodels.PixelCNN(64, (32, 32)).to(device)

tfms = TF.Compose([
        TF.Resize(32),
        TF.ToTensor(),
])

dl = DataLoader(MNIST('.', transform=tfms), batch_size=16, shuffle=True,
        num_workers=4)

opt = Adam(pixcnn.parameters(), lr=2e-4)

iters = 0
while True:
    for x, y in dl:
        x = x.expand(-1, 3, -1, -1)
        x = x.to(device)

        opt.zero_grad()
        x2 = pixcnn(x * 2 - 1)
        loss = F.cross_entropy(x2, (x * 255).long())
        loss.backward()
        opt.step()
        print(x2[0, :, 0].argmax(1))
        print(loss)
        vis.images(x2.argmax(dim=1) / 255.0, win='recon')

        if iters % 30 == 0:
            with torch.no_grad():
                vis.images(pixcnn.sample(0.3, 4), win='new')

        iters += 1
