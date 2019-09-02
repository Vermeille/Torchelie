import torchelie.datasets.debug
from .concat import HorizontalConcatDataset

import torch


class PairedDataset(torch.utils.data.Dataset):
    """
    A dataset that returns all possible pairs of samples of two datasets

    Args:
        dataset1 (Dataset): a dataset
        dataset2 (Dataset): another dataset
    """
    def __init__(self, dataset1, dataset2):
        super(PairedDataset, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, i):
        idx1 = i % len(self.dataset1)
        idx2 = i // len(self.dataset2)

        x1 = self.dataset1[idx1]
        x2 = self.dataset2[idx2]

        return zip(x1, x2)

    def __len__(self):
        return len(self.dataset1) * len(self.dataset2)


def mixup(x1, x2, y1, y2, num_classes, mixer=None, alpha=0.4):
    r"""
    Mixes samples `x1` and `x2` with respective labels `y1` and `y2` according
    to MixUp

    :math:`\lambda \sim \text{Beta}(\alpha, \alpha)`

    :math:`x = \lambda x_1 + (1-\lambda) x_2`

    :math:`y = \lambda y_1 + (1 - \lambda) y_2`
    """
    if mixer is None:
        alpha = torch.tensor([alpha])
        mixer = torch.distributions.Beta(alpha, alpha)

    lam = mixer.sample(y1.shape).to(y1.device)
    y1 = torch.nn.functional.one_hot(
        torch.tensor(y1), num_classes=num_classes).float().to(y1.device)
    y2 = torch.nn.functional.one_hot(
        torch.tensor(y2), num_classes=num_classes).float().to(y1.device)

    return (lam * x1 + (1 - lam) * x2), (lam * y1 + (1 - lam) * y2)


class MixUpDataset(PairedDataset):
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
        super(MixUpDataset, self).__init__(dataset, dataset)
        alpha = torch.tensor([alpha])
        self.mixer = torch.distributions.Beta(alpha, alpha)

    def __getitem__(self, i):
        (x1, x2), (y1, y2) = super(MixUpDataset, self).__getitem__(i)

        mixer = torch.distributions.Beta(alpha, alpha)
        return mixup(x1, x2, y1, y2, len(self.dataset1.classes), mixer)


class NoexceptDataset:
    """
    Wrap a dataset and absorbs the exceptions it raises.  Useful in case of a
    big downloaded dataset with corrupted samples for instance.

    Args:
        ds (Dataset): a dataset
    """
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        try:
            return self.ds[i]
        except Exception as e:
            print(e)
            if i < len(self):
                return self[i + 1]
            else:
                return self[0]
