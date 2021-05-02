"""
Train a classifier.

It logs train and test loss and accuracy. It also displays confusion matrix and
gradient based feature visualization throughout training.

If you have an image training directory with a folder per class and a testing
directory with a folder per class, you can run it with a command line.

`python3 -m torchelie.recipes.classification --trainset path/to/train --testset
path/to/test`
"""
import torch
import torchvision.models as tvmodels

import torchelie.callbacks as tcb
import torchelie.utils as tu
from torchelie.recipes.trainandtest import TrainAndTest
from torchelie.optim import RAdamW


def Classification(model,
                   train_fun,
                   test_fun,
                   train_loader,
                   test_loader,
                   classes,
                   *,
                   visdom_env=None,
                   checkpoint=None,
                   test_every=1000,
                   log_every=100):
    """
    Classification training and testing loop. Both forward functions must
    return a per-batch loss and logits predictions. It expands from
    :code:`TrainAndTest`. Both :code:`train_fun` and :code:`test_fun` must
    :code:`return {'loss': batch_loss, 'preds': logits_predictions}`. The model
    is automatically registered and checkpointed as :code:`checkpoint['model']`,
    and put in eval mode when testing. The list of classes is checkpointed as
    well in :code:`checkpoint['classes']`.


    Training callbacks:

    - AccAvg for displaying accuracy
    - WindowedMetricAvg for displaying loss
    - ConfusionMatrix if len(classes) <= 25
    - ClassificationInspector
    - MetricsTable

    Inherited training callbacks:

    - Counter for counting iterations, connected to the testing loop as well
    - VisdomLogger
    - StdoutLogger

    Testing:

    Testing loop is in :code:`.test_loop`.

    Testing callbacks:

    - AccAvg
    - WindowedMetricAvg
    - ConfusionMatrix if :code:`len(classes) <= 25`
    - ClassificationInspector
    - MetricsTable

    Inherited testing callbacks:

    - VisdomLogger
    - StdoutLogger
    - Checkpoint saving the best testing loss


    Args:
        model (nn.Model): a model
        train_fun (Callabble): a function that takes a batch as a single
            argument, performs a training forward pass and return a dict of
            values to populate the recipe's state. It expects the logits
            predictions under key "preds", and this batch's loss under key
            "loss".
        test_fun (Callable): a function taking a batch as a single argument
            then performs something to evaluate your model and returns a dict
            to populate the state. It expects the logits
            predictions under key "preds", and this batch's loss under key
            "loss".
        train_loader (DataLoader): Training set dataloader
        test_loader (DataLoader): Testing set dataloader
        classes (list of str): classes name, in order
        visdom_env (str): name of the visdom environment to use, or None for
            not using Visdom (default: None)
        test_every (int): testing frequency, in number of iterations (default:
            1000)
        log_every (int): logging frequency, in number of iterations (default:
            100)
    """

    key_best = (lambda state: -state['test_loop']['callbacks']['state']
                ['metrics']['loss'])

    loop = TrainAndTest(model,
                        train_fun,
                        test_fun,
                        train_loader,
                        test_loader,
                        visdom_env=visdom_env,
                        test_every=test_every,
                        log_every=log_every,
                        checkpoint=checkpoint,
                        key_best=key_best)

    loop.callbacks.add_callbacks([
        tcb.AccAvg(),
        tcb.WindowedMetricAvg('loss'),
    ])
    loop.register('classes', classes)

    loop.test_loop.callbacks.add_callbacks([
        tcb.AccAvg(post_each_batch=False, avg_type='running'),
        tcb.WindowedMetricAvg('loss', False),
    ])

    if visdom_env is not None:
        if len(classes) <= 50:
            loop.callbacks.add_epilogues([
                tcb.ConfusionMatrix(classes),
            ])
        loop.callbacks.add_epilogues([
            tcb.ClassificationInspector(30, classes),
            tcb.MetricsTable()
        ])

    if len(classes) <= 50:
        loop.test_loop.callbacks.add_callbacks([
            tcb.ConfusionMatrix(classes),
        ])

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
                               beta1=0.9,
                               wd=1e-2,
                               visdom_env='main',
                               test_every=1000,
                               log_every=100,
                               checkpoint='model'):
    """
    Extends Classification with default cross entropy forward passes. Also adds
    RAdamW and ReduceLROnPlateau.

    Inherited training callbacks:

    - AccAvg for displaying accuracy
    - WindowedMetricAvg for displaying loss
    - ConfusionMatrix if len(classes) <= 25
    - ClassificationInspector
    - MetricsTable
    - Counter for counting iterations, connected to the testing loop as well
    - VisdomLogger
    - StdoutLogger

    Training callbacks:

    - Optimizer with RAdamW
    - LRSched with ReduceLROnPlateau

    Testing:

    Testing loop is in :code:`.test_loop`.

    Inherited testing callbacks:

    - AccAvg
    - WindowedMetricAvg
    - ConfusionMatrix if :code:`len(classes) <= 25`
    - ClassificationInspector
    - MetricsTable
    - VisdomLogger
    - StdoutLogger
    - Checkpoint saving the best testing loss

    Args:
        model (nn.Module): a model learnable with cross entropy
        train_loader (DataLoader): Training set dataloader
        test_loader (DataLoader): Testing set dataloader
        classes (list of str): classes name, in order
        lr (float): the learning rate
        beta1 (float): RAdamW's beta1
        wd (float): weight decay
        visdom_env (str): name of the visdom environment to use, or None for
            not using Visdom (default: None)
        test_every (int): testing frequency, in number of iterations (default:
            1000)
        log_every (int): logging frequency, in number of iterations (default:
            1000)
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
                          log_every=log_every,
                          checkpoint=checkpoint)

    opt = RAdamW(model.parameters(),
                 lr=lr,
                 betas=(beta1, 0.999),
                 weight_decay=wd)
    loop.callbacks.add_callbacks([
        tcb.Optimizer(opt, log_lr=True),
        tcb.LRSched(torch.optim.lr_scheduler.ReduceLROnPlateau(opt))
    ])
    return loop


def MixupClassification(model,
                        train_loader,
                        test_loader,
                        classes,
                        *,
                        lr=3e-3,
                        beta1=0.9,
                        wd=1e-2,
                        visdom_env='main',
                        test_every=1000,
                        log_every=100):
    """
    A Classification recipe with a default froward training / testing pass
    using cross entropy and mixup, and extended with RAdamW and
    ReduceLROnPlateau.

    Args:
        model (nn.Module): a model learnable with cross entropy
        train_loader (DataLoader): Training set dataloader. Must have soft
            targets. Should be a DataLoader loading a MixupDataset or
            compatible.
        test_loader (DataLoader): Testing set dataloader. Dataset must have
            categorical targets.
        classes (list of str): classes name, in order
        lr (float): the learning rate
        beta1 (float): RAdamW's beta1
        wd (float): weight decay
        visdom_env (str): name of the visdom environment to use, or None for
            not using Visdom (default: None)
        test_every (int): testing frequency, in number of iterations (default:
            1000)
        log_every (int): logging frequency, in number of iterations (default:
            1000)
    """

    from torchelie.loss import continuous_cross_entropy

    def train_step(batch):
        x, y = batch
        pred = model(x)
        loss = continuous_cross_entropy(pred, y)
        loss.backward()
        return {'loss': loss}

    def validation_step(batch):
        x, y = batch
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        return {'loss': loss, 'pred': pred}

    loop = TrainAndTest(model,
                        train_step,
                        validation_step,
                        train_loader,
                        test_loader,
                        visdom_env=visdom_env,
                        test_every=test_every,
                        log_every=log_every)

    loop.callbacks.add_callbacks([
        tcb.WindowedMetricAvg('loss'),
    ])
    loop.register('classes', classes)

    loop.test_loop.callbacks.add_callbacks([
        tcb.AccAvg(post_each_batch=False),
        tcb.WindowedMetricAvg('loss', False),
    ])

    if visdom_env is not None:
        loop.callbacks.add_epilogues([
            tcb.MetricsTable()
        ])

    if len(classes) <= 25:
        loop.test_loop.callbacks.add_callbacks([
            tcb.ConfusionMatrix(classes),
        ])

    loop.test_loop.callbacks.add_callbacks([
        tcb.ClassificationInspector(30, classes, False),
        tcb.MetricsTable(False)
    ])

    opt = RAdamW(model.parameters(),
                 lr=lr,
                 betas=(beta1, 0.999),
                 weight_decay=wd)
    loop.callbacks.add_callbacks([
        tcb.Optimizer(opt, log_lr=True),
        tcb.LRSched(torch.optim.lr_scheduler.ReduceLROnPlateau(opt))
    ])
    return loop


if __name__ == '__main__':
    import argparse

    from torchelie.datasets import CachedDataset
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    import torchvision.transforms as TF
    import torchelie.transforms as TTF

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset', type=str, required=True)
    parser.add_argument('--testset', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--im-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--visdom-env', type=str)
    parser.add_argument('--no-cache', action='store_false')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--mixup', action='store_true')
    args = parser.parse_args()

    tfm = TF.Compose([
        TF.RandomResizedCrop(args.im_size, (0.7, 1.1)),
        TF.ColorJitter(0.5, 0.5, 0.4, 0.05),
        TF.RandomHorizontalFlip(),
        TF.ToTensor(),
        TF.Normalize([0.5] * 3, [0.5] * 3, True),
    ])
    tfm_test = TF.Compose([
        TTF.ResizedCrop(args.im_size, scale=1),
        TF.Resize(args.im_size),
        TF.ToTensor(),
        TF.Normalize([0.5] * 3, [0.5] * 3, True)
    ])

    if args.no_cache:
        trainset = ImageFolder(args.trainset, transform=tfm)
        testset = ImageFolder(args.testset, transform=tfm_test)
    else:
        trainset = ImageFolder(args.trainset)
        testset = ImageFolder(args.testset)

        trainset = CachedDataset(trainset, transform=tfm)
        testset = CachedDataset(testset, transform=tfm_test)

    if args.mixup:
        from torchelie.datasets import MixUpDataset
        trainset = MixUpDataset(trainset)

    trainloader = DataLoader(trainset,
                             args.batch_size,
                             num_workers=8,
                             pin_memory=True,
                             shuffle=True,
                             drop_last=True)
    testloader = DataLoader(testset,
                            args.batch_size,
                            num_workers=8,
                            pin_memory=True)

    model = tvmodels.resnet18(pretrained=False)
    model.fc = tu.kaiming(torch.nn.Linear(512, len(testset.classes)))

    if args.mixup:
        clf_recipe = MixupClassification(model,
                                         trainloader,
                                         testloader,
                                         testset.classes,
                                         log_every=10,
                                         test_every=50,
                                         lr=args.lr,
                                         beta1=args.beta1,
                                         wd=args.wd,
                                         visdom_env=args.visdom_env)
    else:
        clf_recipe = CrossEntropyClassification(model,
                                                trainloader,
                                                testloader,
                                                testset.classes,
                                                log_every=10,
                                                test_every=50,
                                                lr=args.lr,
                                                beta1=args.beta1,
                                                wd=args.wd,
                                                visdom_env=args.visdom_env)

    clf_recipe.to(args.device)
    clf_recipe.run(args.epochs)
