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


class MixMatchDataset(PairedDataset):
    """
    https://arxiv.org/abs/1905.02249
    """
    def __init__(self, dataset, alpha=0.4):
        super(MixMatchDataset, self).__init__(dataset, dataset)
        alpha = torch.tensor([alpha])
        self.mixer = torch.distributions.Beta(alpha, alpha)

    def __getitem__(self, i):
        (x1, x2), (y1, y2) = super(MixMatchDataset, self).__getitem__(i)

        y1 = torch.nn.functional.one_hot(torch.tensor(y1),
                                         num_classes=len(
                                             self.dataset1.classes)).float()
        y2 = torch.nn.functional.one_hot(torch.tensor(y2),
                                         num_classes=len(
                                             self.dataset2.classes)).float()
        lam = self.mixer.sample()

        return (lam * x1 + (1 - lam) * x2), (lam * y1 + (1 - lam) * y2)
