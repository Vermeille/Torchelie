from torchelie.datasets import *

class StupidDataset:
    def __init__(self, cls):
        self.classes = [0, 1]
        self.cls = cls

    def __len__(self):
        return 10

    def __getitem__(self, i):
        return torch.FloatTensor([i]), torch.LongTensor([self.cls])

def test_colored():
    cc = ColoredColumns(64, 64)
    cr = ColoredRows(64, 64)

    cc[0]
    cr[0]


def test_paired():
    ds = StupidDataset(0)
    ds2 = StupidDataset(1)

    pd = PairedDataset(ds, ds2)
    assert len(pd) == 100
    assert pd[0] == list(zip(ds[0], ds2[0]))


def test_mixup():
    ds = StupidDataset(0)
    md = MixUpDataset(ds)
    md[0]

