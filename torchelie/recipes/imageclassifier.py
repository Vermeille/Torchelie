import torch
import torch.optim as optim
import torchvision.models as tvmodels

import torchelie.metrics.callbacks as cb


class ImageClassifier:
    def __init__(self,
                 train_loader,
                 test_loader,
                 test_every=1000,
                 train_callbacks=[
                     cb.WindowedLossAvg(),
                     cb.AccAvg(),
                     cb.LogInput(),
                     cb.VisdomLogger(visdom_env='main', log_every=100)
                 ],
                 test_callbacks=[
                     cb.EpochLossAvg(False),
                     cb.AccAvg(False),
                     cb.VisdomLogger(visdom_env='main', log_every=-1,
                     prefix='test_')
                 ],
                 device='cpu',
                 **kwargs):
        self.device = device
        num_classes = len(train_loader.dataset.classes)
        self.model = tvmodels.vgg11_bn(num_classes=num_classes).to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=3e-5)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_every = test_every
        self.state = {'metrics': {}}
        self.test_state = {'metrics': {}}
        self.train_callbacks = train_callbacks
        self.test_callbacks = test_callbacks

    def forward(self, x, y):
        self.opt.zero_grad()
        pred = self.model(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        loss.backward()
        self.opt.step()
        return loss, pred

    def _run_train_cbs(self, trigger):
        for cb in self.train_callbacks:
            if hasattr(cb, trigger):
                getattr(cb, trigger)(self.state)

    def _run_test_cbs(self, trigger):
        for cb in self.test_callbacks:
            if hasattr(cb, trigger):
                getattr(cb, trigger)(self.test_state)

    def evaluate(self):
        self._run_test_cbs('on_epoch_start')
        for x_cpu, y_cpu in self.test_loader:
            self.test_state['batch'] = (x_cpu, y_cpu)
            self._run_test_cbs('on_batch_start')

            x = x_cpu.to(self.device, non_blocking=True)
            y = y_cpu.to(self.device, non_blocking=True)

            loss, pred = self.forward(x, y)
            loss = loss.cpu().detach()
            pred = pred.cpu().detach()
            self.test_state['loss'] = loss
            self.test_state['pred'] = pred

            self._run_test_cbs('on_batch_end')

        self._run_test_cbs('on_epoch_end')

    def __call__(self, epochs=5):
        self.iters = 0
        for epoch in range(epochs):
            self._run_train_cbs('on_epoch_start')
            for x_cpu, y_cpu in self.train_loader:
                self.state['batch'] = (x_cpu, y_cpu)
                self._run_test_cbs('on_batch_start')
                x = x_cpu.to(self.device, non_blocking=True)
                y = y_cpu.to(self.device, non_blocking=True)

                loss, pred = self.forward(x, y)
                loss = loss.cpu().detach()
                pred = pred.cpu().detach()
                self.state['loss'] = loss
                self.state['pred'] = pred

                self._run_train_cbs('on_batch_end')

                if self.iters % self.test_every == 0:
                    self.model.eval()
                    self.evaluate()
                    self.model.train()
                self.iters += 1
            self._run_train_cbs('on_epoch_end')

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

    trainloader = DataLoader(trainset,
                             32,
                             num_workers=4,
                             pin_memory=True,
                             shuffle=True)
    testloader = DataLoader(testset,
                            32,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=True)
    clf_recipe = ImageClassifier(trainloader,
                                 testloader,
                                 device='cuda',
                                 visdom_env='clf')
    clf_recipe()
