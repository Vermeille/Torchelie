import torch
import torch.nn.functional as F
import torchelie.models as tmodels
from torchelie.metrics.avg import WindowAvg

import torchvision.transforms as TF
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torch.utils.data import DataLoader
from torch.optim import Adam

from visdom import Visdom

device = 'cuda'

vis = Visdom(env='pixcnn')

pixcnn = tmodels.PixelCNN(64, (32, 32), channels=3).to(device)

tfms = TF.Compose([
        TF.Resize(32),
        TF.ToTensor(),
])


dl = DataLoader(
        FashionMNIST('.', transform=tfms, download=True),
        batch_size=8, shuffle=True,
        num_workers=4)

opt = Adam(pixcnn.parameters(), lr=3e-3, betas=(0.95, 0.9995))


sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5000,
        factor=0.8, cooldown=10000, min_lr=1e-6)

avg_loss = WindowAvg()
iters = 0
while True:
    for x, y in dl:
        x = x.expand(-1, 3, -1, -1)
        x = x.to(device)

        opt.zero_grad()
        x2 = pixcnn(x * 2 - 1)
        loss = F.cross_entropy(x2, (x * 255).long())
        loss.backward()
        avg_loss.log(loss.item())
        opt.step()
        sched.step(avg_loss.get())
        if iters % 10 == 0:
            print(avg_loss.get())
            vis.images(x2.argmax(dim=1).float() / 255.0, win='recon')
            vis.line(X=[iters], Y=[avg_loss.get()], win='loss',
                    update='append')
            vis.line(X=[iters], Y=[opt.param_groups[0]['lr']], win='lr',
                    update='append')

        if iters % 1000 == 0:
            print('sample')
            pixcnn.eval()
            vis.images(pixcnn.sample(1, 4).expand(-1, 3, -1, -1), win='new')
            pixcnn.train()
        if iters % 1000 == 0:
            print('sample')
            pixcnn.eval()
            vis.images(pixcnn.partial_sample(x[:4], 0.1).expand(-1, 3,
                -1, -1), win='partial')
            pixcnn.train()

        iters += 1
