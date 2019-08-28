import torchelie.datasets.debug
from .concat import HorizontalConcatDataset

import torch


class PairedDataset(torch.utils.data.Dataset):
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
    https://arxiv.org/abs/1905.02249
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
