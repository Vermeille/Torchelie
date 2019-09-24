import torch
import torch.nn.functional as F
import torchelie.models as tmodels
from torchelie.metrics.avg import WindowAvg

import torchvision.transforms as TF
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torch.utils.data import DataLoader
from torchelie.optim import RAdamW
from torchelie.recipes import TrainAndCall


class PixMe(torch.nn.Module):
    def __init__(self):
        super(PixMe, self).__init__()
        self.model = tmodels.PixelCNN(64, (32, 32), channels=3)

    def make_optimizer(self):
        return RAdamW(self.model.parameters(), lr=3e-3, betas=(0.95, 0.9995))

    def train_step(self, batch, opt):
        x = batch[0]
        x = x.expand(-1, 3, -1, -1)

        opt.zero_grad()
        x2 = self.model(x * 2 - 1)
        loss = F.cross_entropy(x2, (x * 255).long())
        loss.backward()
        opt.step()

        reconstruction = x2.argmax(dim=1).float() / 255.0
        return {'loss': loss, 'metrics': {'reconstruction': reconstruction}}

    def after_train(self):
        imgs = self.model.sample(1, 4).expand(-1, 3, -1, -1)
        return {'metrics': {'imgs': imgs}}


tfms = TF.Compose([
    TF.Resize(32),
    TF.ToTensor(),
])

dl = DataLoader(FashionMNIST('~/.cache/torch/fashionmnist',
                             transform=tfms,
                             download=True),
                batch_size=8,
                shuffle=True,
                num_workers=4)

trainer = TrainAndCall(PixMe(), visdom_env='pixelcnn', device='cuda')
trainer(dl, 10)
