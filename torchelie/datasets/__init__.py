import random
import multiprocessing

import torchelie.utils as tu
from torchelie.datasets.debug import *
from .concat import HorizontalConcatDataset, MergedDataset
from .ms1m import MS1M
from .pix2pix import Pix2PixDataset, ImagesPaths, UnlabeledImages
from .pix2pix import SideBySideImagePairsDataset

from typing import Sequence
from torchvision.datasets import ImageFolder
from pathlib import Path
import os
import torch


def FastImageFolder(root: str, *args, **kwargs) -> ImageFolder:
    """
    Load an ImageFolder dataset faster by caching the file list the first
    time it is accessed. Force refreshing by deleting
    :code:`{root}/cached_list.pth`.

    Forwards all arguments to torchvision.datasets.ImageFolder

    Returns:
        The ImageFolder transparently loaded faster.
    """
    cache = Path(root) / 'cached_list.pth'
    fields = ['samples', 'imgs', 'classes', 'class_to_idx']
    if cache.exists():
        os.makedirs('/tmp/empty/0', exist_ok=True)
        open('/tmp/empty/0/0.jpg', 'a').close()
        ds = ImageFolder('/tmp/empty', *args, **kwargs)
        save = torch.load(str(cache))
        for f in fields:
            setattr(ds, f, save[f])
    else:
        ds = ImageFolder(root, *args, **kwargs)
        torch.save({f: getattr(ds, f) for f in fields}, str(cache))
    return ds


class PairedDataset(torch.utils.data.Dataset):
    """
    A dataset that returns all possible pairs of samples of two datasets

    Args:
        dataset1 (Dataset): a dataset
        dataset2 (Dataset): another dataset
    """
    def __init__(self, dataset1: Sequence, dataset2: Sequence) -> None:
        super(PairedDataset, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, i):
        idx1 = i % len(self.dataset1)
        idx2 = i // len(self.dataset2)

        x1 = self.dataset1[idx1]
        x2 = self.dataset2[idx2]

        if isinstance(x1, (tuple, list)):
            return list(zip(x1, x2))
        return x1, x2

    def __len__(self) -> int:
        return len(self.dataset1) * len(self.dataset2)

    def __repr__(self) -> str:
        return ("PairedDataset:\n" + tu.indent(repr(self.dataset1))
                + "\n--\n" + tu.indent(repr(self.dataset2)))


class RandomPairsDataset(torch.utils.data.Dataset):
    """
    :code:`PairedDataset` deterministically constructs all possible pairs of
    items from both datasets. This may cause some problems while sampling
    if permutation or weight array must be constructed as they would be of size
    :code:`len(dataset1) * len(dataset2)` and may not fit in memory.

    :code:`RandomPairsDataset` presents itself as dataset of 10000 elements,
    and accessing the ith item returns a random pair, effectively ignoring i.

    Args:
        dataset1 (Dataset): a dataset
        dataset2 (Dataset): another dataset
    """
    def __init__(self, dataset1, dataset2):
        super(RandomPairsDataset, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, i):
        idx1 = random.randrange(0, len(self.dataset1) - 1)
        idx2 = random.randrange(0, len(self.dataset2) - 1)

        x1 = self.dataset1[idx1]
        x2 = self.dataset2[idx2]

        if isinstance(x1, (tuple, list)):
            return list(zip(x1, x2))
        return x1, x2

    def __len__(self):
        return 10000

    def __repr__(self):
        return ("PairedDataset:\n" + tu.indent(repr(self.dataset1))
                + "\n--\n" + tu.indent(repr(self.dataset2)))


def mixup(x1, x2, y1, y2, num_classes, mixer=None, alpha=0.4):
    r"""
    Mixes samples `x1` and `x2` with respective labels `y1` and `y2` according
    to MixUp

    :math:`\lambda \sim \text{Beta}(\alpha, \alpha)`

    :math:`x = \lambda x_1 + (1-\lambda) x_2`

    :math:`y = \lambda y_1 + (1 - \lambda) y_2`

    Args:
        x1 (tensor): sample 1
        x2 (tensor): sample 2
        y1 (tensor): label 1
        y2 (tensor): label 2
        num_classes (int): number of classes
        mixer (Distribution, optional): a distribution to sample lambda from.
            If unspecified, the distribution will be a Beta(alpha, alpha)
        alpha (float): if mixer is unspecified, used to parameterize the Beta
            distribution
    """
    if mixer is None:
        alpha = torch.tensor([alpha])
        mixer = torch.distributions.Beta(alpha, alpha)

    y1 = torch.tensor(y1)
    y2 = torch.tensor(y2)
    lam = mixer.sample(y1.shape).to(y1.device)
    y1 = torch.nn.functional.one_hot(y1, num_classes=num_classes).float().to(
        y1.device)
    y2 = torch.nn.functional.one_hot(y2, num_classes=num_classes).float().to(
        y1.device)

    return (lam * x1 + (1 - lam) * x2), (lam * y1 + (1 - lam) * y2)


class _Wrap:
    def __init__(self, instance):
        # FIXME: NOT WORKING WHEN SETTING MEMBERS
        # self.__dict__ = instance.__dict__
        self.ds = instance

    def __getattr__(self, attr):
        return getattr(self.ds, attr)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class MixUpDataset(_Wrap):
    """
    Linearly mixes two samples and labels from a dataset according to the MixUp
    algorithm

    https://arxiv.org/abs/1905.02249

    Args:
        dataset (Dataset): the dataset
        alpha (float): the alpha that parameterizes the beta distribution from
            which the blending factor is sampled
    """
    def __init__(self, dataset, alpha=0.4):
        super(MixUpDataset, self).__init__(dataset)
        self.ds = dataset
        alpha = torch.tensor([alpha])
        self.mixer = torch.distributions.Beta(alpha, alpha)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x1, y1 = self.ds[i]
        x2, y2 = random.choice(self.ds)

        return mixup(x1, x2, y1, y2, len(self.ds.classes), self.mixer)

    def __repr__(self):
        return f"MixUpDataset({self.mixer}):\n" + tu.indent(repr(self.ds))


class _Proxy:
    def __init__(self, ds, indices, remap_unused_classes, cls_map):
        self.ds = ds
        self.indices = indices
        self.remap_unused_classes = remap_unused_classes
        self.cls_map = cls_map

    def __len__(self):
        return len(indices)

    def __getitem__(self, i):
        b = self.ds.samples[self.indices[i]]
        if self.remap_unused_classes:
            b = list(b)
            b[1] = self.cls_map[b[1]]
        return b


class Subset:
    """
    Create a subset that is a random ratio of a dataset.

    Args:
        ds (Dataset): the dataset to sample from. Must have a :code:`.samples`
            member like torchvision's datasets.
        ratio (float): a value between 0 and 1, the subsampling ratio.
        remap_unused_classes (boolean): if True, classes not represented in the
            subset will not be considered. Remaining classes will be numbered
            from 0 to N.
    """
    def __init__(self, ds, ratio, remap_unused_classes=False):
        self.ratio = ratio
        self.ds = ds
        indices = [i for i in range(len(ds)) if random.uniform(0, 1) < ratio]
        self.indices = indices

        self.remap_classes = remap_unused_classes
        if remap_unused_classes:
            cls_map = {}
            cls = []
            cls_to_idx = {}
            for i in self.indices:
                c = ds.samples[i][1]
                if c not in cls_map:
                    new_idx = len(cls_map)
                    cls_map[c] = new_idx
                    cls.append(ds.classes[c])
                    cls_to_idx[ds.classes[c]] = new_idx
            self.cls_map = cls_map
            self.classes = cls
            self.class_to_idx = cls_to_idx
        else:
            cls_map = {}
            self.classes = ds.classes
            self.class_to_idx = ds.class_to_idx

        self.samples = _Proxy(ds, indices, remap_unused_classes, cls_map)
        self.imgs = self.samples

    def __repr__(self):
        return "Subset(len={}, n_classes={}):\n{}".format(
            len(self.indices), len(self.classes), tu.indent(repr(self.ds)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        b = self.ds[self.indices[i]]
        if self.remap_classes:
            b = list(b)
            b[1] = self.cls_map[b[1]]
        return b


class NoexceptDataset(_Wrap):
    """
    Wrap a dataset and absorbs the exceptions it raises.  Useful in case of a
    big downloaded dataset with corrupted samples for instance.

    Args:
        ds (Dataset): a dataset
    """
    def __init__(self, ds):
        super(NoexceptDataset, self).__init__(ds)
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        while True:
            try:
                return self.ds[i]
            except Exception as e:
                print(e)
            if i < len(self) - 1:
                i += 1
            else:
                i = 0

    def __repr__(self):
        return "NoexceptDataset:\n{}".format(tu.indent(repr(self.ds)))


class WithIndexDataset(_Wrap):
    """
    Wrap a dataset. Also returns the index of the accessed element. Original
    dataset's attributes are transparently accessible

    Args:
        ds (Dataset): A dataset
    """
    def __init__(self, ds):
        super(WithIndexDataset, self).__init__(ds)
        self.ds = ds

    def __getitem__(self, i):
        """
        Args:
            i (int): index

        Returns:
            A tuple (i, self.ds[i])
        """
        return i, self.ds[i]

    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return "WithIndexDataset:\n{}".format(tu.indent(repr(self.ds)))


class CachedDataset(_Wrap):
    """
    Wrap a dataset. Lazily caches *in RAM* elements returned by the underlying
    dataset.

    Args:
        ds (Dataset): A dataset
        transform (Callable): transform to apply on cached elements
        device: the device on which the cache is allocated
    """
    def __init__(self, ds, transform=None, device='cpu'):
        super(CachedDataset, self).__init__(ds)
        self.ds = ds
        self.transform = transform
        self.cache = multiprocessing.Manager().list([None] * len(self.ds))
        self.device = device

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        """
        Returns:
            The ith element of the underlying dataset or its cached value if
            available
        """
        if self.cache[i] is None:
            self.cache[i] = tu.send_to_device(self.ds[i],
                                              self.device,
                                              non_blocking=True)

        x, *y = self.cache[i]

        if self.transform is not None:
            x = self.transform(x)

        return [x] + y

    def __repr__(self):
        return "CachedDataset:\n{}".format(tu.indent(repr(self.ds)))
