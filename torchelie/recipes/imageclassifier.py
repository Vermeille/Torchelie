import torch
import torch.optim as optim
import torchvision.models as tvmodels

from torchelie.metrics import WindowAvg, RunningAvg
from torchelie.recipes.recipebase import RecipeBase


class ImageClassifier(RecipeBase):
    def __init__(self,
                 train_loader,
                 test_loader,
                 test_every=1000,
                 device='cpu',
                 **kwargs):
        super(ImageClassifier, self).__init__(**kwargs)
        self.device = device
        num_classes = len(train_loader.dataset.classes)
        self.model = tvmodels.vgg11_bn(num_classes=num_classes).to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=3e-5)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_every = test_every

    def forward(self, x, y):
        self.opt.zero_grad()
        pred = self.model(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        loss.backward()
        self.opt.step()
        return loss, pred

    def evaluate(self):
        loss_avg = RunningAvg()
        acc_avg = RunningAvg()
        for x_cpu, y_cpu in self.test_loader:
            x = x_cpu.to(self.device, non_blocking=True)
            y = y_cpu.to(self.device, non_blocking=True)

            loss, pred = self.forward(x, y)
            loss = loss.cpu()
            pred = pred.cpu()
            correct = pred.detach().argmax(1).eq(y_cpu).float().sum()

            loss_avg.log(loss.item())
            acc_avg.log(correct, pred.shape[0])

        self.log(
            {
                'test_x': x_cpu,
                'test_loss': loss_avg.get(),
                'test_accuracy': acc_avg.get()
            },
            force=True)

    def __call__(self, epochs=5):
        self.iters = 0
        loss_avg = WindowAvg(k=self.log_every * 2)
        for epoch in range(epochs):
            acc_avg = RunningAvg()
            for x_cpu, y_cpu in self.train_loader:
                x = x_cpu.to(self.device, non_blocking=True)
                y = y_cpu.to(self.device, non_blocking=True)

                loss, pred = self.forward(x, y)
                loss = loss.cpu()
                pred = pred.cpu()
                correct = pred.detach().argmax(1).eq(y_cpu).float().sum()

                loss_avg.log(loss.item())
                acc_avg.log(correct, pred.shape[0])

                self.log({
                    'train_x': x_cpu,
                    'train_loss': loss_avg.get(),
                    'train_accuracy': acc_avg.get()
                })

                if self.iters % self.test_every == 0:
                    self.model.eval()
                    self.evaluate()
                    self.model.train()
                self.iters += 1

        self.model.eval()
        return self.model


if __name__ == '__main__':
    from torchvision.datasets import FashionMNIST
    from torch.utils.data import DataLoader
    import torchvision.transforms as TF
    tfm = TF.Compose([
        TF.Resize(128),
        TF.Grayscale(3),
        TF.ToTensor(),
    ])
    trainset = FashionMNIST('../tests/', transform=tfm)
    testset = FashionMNIST('../tests/', train=False, transform=tfm)

    trainloader = DataLoader(trainset, 32, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, 32, num_workers=4, pin_memory=True)
    clf_recipe = ImageClassifier(trainloader,
                                 testloader,
                                 device='cuda',
                                 visdom_env='clf')
    clf_recipe()
