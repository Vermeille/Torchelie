import torch
import torch.nn.functional as F
import torchelie.models as tmodels

import torchvision.transforms as TF
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torch.utils.data import DataLoader
from torch.optim import Adam

from visdom import Visdom

device = 'cuda'

vis = Visdom(env='pixcnn')

pixcnn = tmodels.PixelCNN(64, (32, 32), channels=1).to(device)

tfms = TF.Compose([
        TF.Resize(32),
        TF.ToTensor(),
])

dl = DataLoader(FashionMNIST('.', transform=tfms, download=True), batch_size=64, shuffle=True,
        num_workers=4)

opt = Adam(pixcnn.parameters(), lr=1e-2)

class RunningAvg:
    def __init__(self):
        self.count = 0
        self.val = 0

    def log(self, x):
        self.count += 1
        self.val += x

    def get(self):
        return self.val / self.count

    def print(self):
        print(self.get(), self.val, '/', self.count)


iters = 0
while True:
    avg_loss = RunningAvg()
    for x, y in dl:
        x = x.mean(dim=1, keepdim=True)
        x = x.to(device)

        opt.zero_grad()
        x2 = pixcnn(x * 2 - 1)
        loss = F.cross_entropy(x2, (x * 255).long())
        loss.backward()
        avg_loss.log(loss.item())
        opt.step()
        avg_loss.print()
        if iters % 10 == 0:
            vis.images(x2.argmax(dim=1).float() / 255.0, win='recon')
            vis.line(X=[iters], Y=[avg_loss.get()], win='loss',
                    update='append')

        if iters % 50 == 0:
            print('sample')
            pixcnn.eval()
            vis.images(pixcnn.sample(1, 4).expand(-1, 3, -1, -1), win='new')
            pixcnn.train()

        iters += 1
