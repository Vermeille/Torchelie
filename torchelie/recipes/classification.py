"""
Train a classifier.

It logs train and test loss and accuracy. It also displays confusion matrix and
gradient based feature visualization throughout training.

If you have an image training directory with a folder per class and a testing
directory with a folder per class, you can run it with a command line.

`python3 -m torchelie.recipes.classification --trainset path/to/train --testset
path/to/test`
"""
from typing import List, Optional, Callable, Iterable, Any

import torch
import torchvision.models as tvmodels

import torchelie as tch
import torchelie.callbacks as tcb
import torchelie.utils as tu
from torchelie.lr_scheduler import FlatAndCosineEnd
from torchelie.transforms.randaugment import RandAugment
from torchelie.recipes.trainandtest import TrainAndTest
from torchelie.optim import RAdamW, Lookahead


def Classification(model,
                   train_fun: Callable,
                   test_fun: Callable,
                   train_loader: Iterable[Any],
                   test_loader: Iterable[Any],
                   classes: List[str],
                   *,
                   visdom_env: Optional[str] = None,
                   checkpoint: Optional[str] = None,
                   test_every: int = 1000,
                   log_every: int = 100):
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

    key_best = (lambda state: -state['test_loop']['callbacks']['state'][
        'metrics']['loss'])

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
        else:
            loop.callbacks.add_callbacks([
                tcb.TopkAccAvg(),
            ])
        loop.callbacks.add_epilogues(
            [tcb.ClassificationInspector(30, classes),
             tcb.MetricsTable()])

    if len(classes) <= 50:
        loop.test_loop.callbacks.add_callbacks([
            tcb.ConfusionMatrix(classes),
        ])
    else:
        loop.test_loop.callbacks.add_callbacks([
            tcb.TopkAccAvg(post_each_batch=False, avg_type='running'),
        ])

    loop.test_loop.callbacks.add_callbacks([
        tcb.ClassificationInspector(30, classes, False),
        tcb.MetricsTable(False)
    ])

    loop.callbacks.add_epilogues([tcb.Throughput()])
    return loop


def CrossEntropyClassification(model,
                               train_loader: Iterable[Any],
                               test_loader: Iterable[Any],
                               classes: List[str],
                               lr: float = 3e-3,
                               beta1: float = 0.9,
                               beta2: float = 0.999,
                               wd: float = 1e-2,
                               visdom_env: Optional[str] = 'main',
                               test_every: int = 1000,
                               log_every: int = 100,
                               checkpoint: Optional[str] = 'model',
                               n_iters: Optional[int] = None):
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
        beta2 (float): RAdamW's beta2
        wd (float): weight decay
        visdom_env (str): name of the visdom environment to use, or None for
            not using Visdom (default: None)
        test_every (int): testing frequency, in number of iterations (default:
            1000)
        log_every (int): logging frequency, in number of iterations (default:
            1000)
        n_iters (optional, int): the number of iterations to train for. If
            provided, switch to a FlatAndCosineEnd scheduler
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

    opt = Lookahead(
        RAdamW(model.parameters(), lr=lr, betas=(beta1, beta2),
               weight_decay=wd))

    loop.register('opt', opt)
    loop.callbacks.add_callbacks([
        tcb.Optimizer(opt, log_lr=True),
    ])
    if n_iters is not None:
        sched = FlatAndCosineEnd(opt, n_iters)
        loop.register('sched', sched)

        loop.callbacks.add_callbacks(
            [tcb.LRSched(sched, step_each_batch=True, metric=None)])
    else:
        loop.callbacks.add_callbacks(
            [tcb.LRSched(torch.optim.lr_scheduler.ReduceLROnPlateau(opt))])
    return loop


def MixupClassification(model,
                        train_loader: Iterable[Any],
                        test_loader: Iterable[Any],
                        classes: List[str],
                        *,
                        lr: float = 3e-3,
                        beta1: float = 0.9,
                        beta2: float = 0.999,
                        wd: float = 1e-2,
                        visdom_env: Optional[str] = 'main',
                        test_every: int = 1000,
                        log_every: int = 100):
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
        beta2 (float): RAdamW's beta2
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
        loop.callbacks.add_epilogues([tcb.MetricsTable()])

    if len(classes) <= 25:
        loop.test_loop.callbacks.add_callbacks([
            tcb.ConfusionMatrix(classes),
        ])

    loop.test_loop.callbacks.add_callbacks([
        tcb.ClassificationInspector(30, classes, False),
        tcb.MetricsTable(False)
    ])

    opt = Lookahead(
        RAdamW(model.parameters(), lr=lr, betas=(beta1, beta2),
               weight_decay=wd))
    loop.register('opt', opt)
    loop.callbacks.add_callbacks([
        tcb.Optimizer(opt, log_lr=True),
        tcb.LRSched(torch.optim.lr_scheduler.ReduceLROnPlateau(opt)),
    ])
    return loop


def train(args, rank, world_size):
    from torchelie.datasets import CachedDataset
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    import torchvision.transforms as TF
    import torchelie.transforms as TTF

    tfm = TF.Compose([
        TF.RandomResizedCrop(args.im_size, (0.3, 1.)),
        TF.RandomHorizontalFlip(),
        #RandAugment(2, 10),
        TF.ToTensor(),
        TF.Normalize([0.5] * 3, [0.5] * 3),
    ])
    tfm_test = TF.Compose([
        TTF.ResizedCrop(args.im_size, scale=1),
        TF.Resize(args.im_size),
        TF.ToTensor(),
        TF.Normalize([0.5] * 3, [0.5] * 3)
    ])

    if not args.cache:
        trainset = tch.datasets.FastImageFolder(args.trainset, transform=tfm)
        testset = tch.datasets.FastImageFolder(args.testset, transform=tfm_test)
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
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=10)

    model = tch.models.preact_resnet18(len(trainset.classes))
    #kkhhmodel = tch.models.preact_resnet20_cifar(len(trainset.classes))
    if rank == 0:
        print('trainset')
        print(trainset)
        print()
        print('testset')
        print(testset)
        print()
        print(model)

    if args.from_ckpt is not None:
        model.load_state_dict(
            torch.load(args.from_ckpt, map_location='cuda:' +
            str(rank))['model'])
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model.to(rank),
                                                          device_ids=[rank],
                                                          output_device=rank)

    if args.mixup:
        clf_recipe = MixupClassification(
            model,
            trainloader,
            testloader,
            testset.classes,
            log_every=50,
            test_every=min(5000,
                           len(trainloader) // 3),
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            wd=args.wd,
            visdom_env=args.visdom_env if rank == 0 else None)
    else:
        clf_recipe = CrossEntropyClassification(
            model,
            trainloader,
            testloader,
            testset.classes,
            log_every=100,
            test_every=min(5000,
                           len(trainloader) // 3),
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            wd=args.wd,
            checkpoint='model' if rank == 0 else None,
            visdom_env=args.visdom_env if rank == 0 else None,
            n_iters=len(trainloader) * args.epochs)

    if rank == 0:
        print(clf_recipe)

    clf_recipe.to(rank)
    clf_recipe.run(args.epochs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset', type=str, required=True)
    parser.add_argument('--testset', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--im-size', type=int, default=64)
    parser.add_argument('--visdom-env', type=str)
    parser.add_argument('--from-ckpt', type=str)
    parser.add_argument('--cache', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--mixup', action='store_true', default=False)
    args = parser.parse_args()

    tu.parallel_run(train, args)
