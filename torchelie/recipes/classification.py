import copy

import torch
import torch.optim as optim
import torchvision.models as tvmodels

import torchelie.metrics.callbacks as cb
import torchelie.utils as tu
from torchelie.recipes.trainandtest import TrainAndTest
from torchelie.optim import RAdamW


class Classification(TrainAndTest):
    """
    Vanilla classification loop and testing loop. Displays Loss and accuracy.

    Args:
        model (nn.Model): a model
            The model must define:

                - `model.make_optimizer()`
                - `model.validation_step(batch)` returns a dict with key loss
                  and pred (raw logits predictions)
                - `model.train_step(batch, opt)` returns a dict with key loss
                  and pred (raw logits predictions)
        visdom_env (str): name of the visdom environment to use, or None for
            not using Visdom (default: None)
        test_every (int): testing frequency, in number of iterations (default:
            1000)
        train_callbacks (list of Callback): additional training callbacks
            (default: [])
        test_callbacks (list of Callback): additional testing callbacks
            (default: [])
        device: a torch device (default: 'cpu')
    """

    def __init__(self,
                 model,
                 visdom_env=None,
                 test_every=1000,
                 train_callbacks=[],
                 test_callbacks=[],
                 device='cpu'):
        super(Classification,
              self).__init__(model=model,
                             visdom_env=visdom_env,
                             test_every=test_every,
                             train_callbacks=[
                                 cb.AccAvg(),
                             ],
                             test_callbacks=[
                                 cb.AccAvg(post_each_batch=False),
                             ],
                             device=device)

    def __call__(self, train_loader, test_loader, epochs=5):
        """
        Runs the recipe.

        Dataloaders must return a batch like (input, target).

        Args:
            train_loader (DataLoader): Training set dataloader
            test_loader (DataLoader): Testing set dataloader
            epochs (int): number of epochs

        Returns:
            trained model, test metrics
        """
        return super(Classification, self).__call__(train_loader, test_loader,
                                                    epochs)


class CrossEntropyLearner(torch.nn.Module):
    """
    Lears a classifier with cross_entropy and adam.

    Args:
        model (nn.Module): the model to be learned.
        lr (float): the learning rate
    """

    def __init__(self, model, lr=3e-5):
        super(CrossEntropyLearner, self).__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def train_step(self, batch, opt):
        x, y = batch
        opt.zero_grad()
        pred = self.forward(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        loss.backward()
        opt.step()
        return {'loss': loss, 'pred': pred}

    def validation_step(self, batch):
        x, y = batch
        pred = self.forward(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        return {'loss': loss, 'pred': pred}

    def make_optimizer(self):
        return RAdamW(self.model.parameters(), lr=self.lr)


class CrossEntropyClassification:
    """
    Lears a classifier with cross_entropy and adam. It just wraps together
    Classification + CrossEntropyLearner

    Args:
        model (nn.Module): a model learnable with cross entropy
        lr (float): the learning rate
        visdom_env (str): name of the visdom environment to use, or None for
            not using Visdom (default: None)
        test_every (int): testing frequency, in number of iterations (default:
            1000)
        train_callbacks (list of Callback): additional training callbacks
            (default: [])
        test_callbacks (list of Callback): additional testing callbacks
            (default: [])
        device: a torch device (default: 'cpu')
    """

    def __init__(self,
                 model,
                 lr=1e-4,
                 visdom_env=None,
                 test_every=1000,
                 train_callbacks=[],
                 test_callbacks=[],
                 device='cpu'):
        self.model = model
        learner = CrossEntropyLearner(model, lr)
        self.clf = Classification(learner,
                                  visdom_env=visdom_env,
                                  test_every=test_every,
                                  train_callbacks=train_callbacks,
                                  test_callbacks=test_callbacks,
                                  device=device)

    def __call__(self, trainloader, testloader, epochs):
        """
        Runs the recipe.

        Dataloaders must return a batch like (input, target).

        Args:
            train_loader (DataLoader): Training set dataloader
            test_loader (DataLoader): Testing set dataloader
            epochs (int): number of epochs

        Returns:
            trained model, test metrics
        """
        _, state = self.clf(trainloader, testloader, epochs)
        return self.model, state


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
                             64,
                             num_workers=4,
                             pin_memory=True,
                             shuffle=True)
    testloader = DataLoader(testset,
                            64,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=True)

    model = tvmodels.vgg11_bn(num_classes=10)

    clf_recipe = Classification(CrossEntropyClassification(model),
                                device='cuda',
                                visdom_env='clf')
    clf_recipe(trainloader, testloader, 4)
