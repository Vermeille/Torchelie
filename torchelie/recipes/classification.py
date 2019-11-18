"""
Train a classifier.

It logs train and test loss and accuracy. (FIXME: add confusion matrix iff
number of classes is tractable)

If you have an image training directory with a folder per class and a testing
directory with a folder per class, you can run it with a command line.

`python3 -m torchelie.recipes.classification --trainset path/to/train --testset
path/to/test`
"""
import copy

import torch
import torch.optim as optim
import torchvision.models as tvmodels

import torchelie.metrics.callbacks as tcb
import torchelie.utils as tu
from torchelie.recipes.trainandtest import TrainAndTest
from torchelie.optim import RAdamW


def Classification(model,
                   train_fun,
                   test_fun,
                   train_loader,
                   test_loader,
                   classes,
                   visdom_env=None,
                   test_every=1000,
                   log_every=100):
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
        train_loader (DataLoader): Training set dataloader
        test_loader (DataLoader): Testing set dataloader
        visdom_env (str): name of the visdom environment to use, or None for
            not using Visdom (default: None)
        test_every (int): testing frequency, in number of iterations (default:
            1000)
        train_callbacks (list of Callback): additional training callbacks
            (default: [])
        test_callbacks (list of Callback): additional testing callbacks
            (default: [])
    """

    loop = TrainAndTest(model,
                        train_fun,
                        test_fun,
                        train_loader,
                        test_loader,
                        visdom_env=visdom_env,
                        test_every=test_every,
                        log_every=log_every)
    loop.callbacks.add_callbacks([
        tcb.AccAvg(),
        tcb.EpochMetricAvg('loss'),
    ])

    loop.test_loop.callbacks.add_callbacks([
        tcb.AccAvg(post_each_batch=False),
        tcb.EpochMetricAvg('loss', False),
    ])

    if visdom_env is not None:
        loop.callbacks.add_epilogues(
            [tcb.ClassificationInspector(30, classes),
             tcb.MetricsTable()])

        loop.test_loop.callbacks.add_callbacks([
            tcb.ClassificationInspector(30, classes, False),
            tcb.MetricsTable(False)
        ])
    return loop


def CrossEntropyClassification(model,
                               train_loader,
                               test_loader,
                               classes,
                               lr=3e-3,
                               visdom_env=None,
                               test_every=1000,
                               log_every=100):
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
    """

    def train_step(batch):
        x, y = batch
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        loss.backward()
        return {'loss': loss, 'pred': pred}

    def validation_step(batch):
        x, y = batch
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        return {'loss': loss, 'pred': pred}

    loop = Classification(model,
                          train_step,
                          validation_step,
                          train_loader,
                          test_loader,
                          classes,
                          visdom_env=visdom_env,
                          test_every=test_every,
                          log_every=log_every)

    opt = RAdamW(model.parameters(), lr=lr)
    loop.callbacks.add_callbacks([
        tcb.Optimizer(opt, log_lr=True),
        tcb.LRSched(torch.optim.lr_scheduler.ReduceLROnPlateau(opt))
    ])
    return loop


if __name__ == '__main__':
    import argparse

    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    import torchvision.transforms as TF

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset', type=str, required=True)
    parser.add_argument('--testset', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--visdom-env', type=str)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    tfm = TF.Compose([
        TF.Resize(128),
        TF.ToTensor(),
    ])
    trainset = ImageFolder(args.trainset, transform=tfm)
    testset = ImageFolder(args.testset, transform=tfm)

    trainloader = DataLoader(trainset,
                             args.batch_size,
                             num_workers=4,
                             pin_memory=True,
                             shuffle=True)
    testloader = DataLoader(testset,
                            args.batch_size,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=True)

    model = tvmodels.resnet18(num_classes=len(trainset.classes))

    clf_recipe = CrossEntropyClassification(model,
                                            trainloader,
                                            testloader,
                                            trainset.classes,
                                            log_every=10,
                                            visdom_env=args.visdom_env)

    clf_recipe.to(args.device)
    clf_recipe.run(args.epochs)
