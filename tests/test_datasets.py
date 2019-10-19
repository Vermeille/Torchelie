from torchelie.datasets import *


class StupidDataset:
    def __init__(self):
        self.classes = [0, 1]
        self.imgs = [(i, 0 if i < 5 else 1) for i in range(10)]
        self.samples = self.imgs

    def __len__(self):
        return 10

    def __getitem__(self, i):
        return torch.FloatTensor([i]), torch.LongTensor([self.imgs[i][1]])


def test_colored():
    cc = ColoredColumns(64, 64)
    cr = ColoredRows(64, 64)

    cc[0]
    cr[0]


def test_paired():
    ds = StupidDataset()
    ds2 = StupidDataset()

    pd = PairedDataset(ds, ds2)
    assert len(pd) == 100
    assert pd[0] == list(zip(ds[0], ds2[0]))


def test_cat():
    ds = StupidDataset()
    ds2 = StupidDataset()

    cated = HorizontalConcatDataset([ds, ds2])
    assert len(cated) == len(ds) * 2
    print([cated[i][1] for i in range(20)])

    for i in range(len(ds)):
        assert cated[i][1] == (0 if i < 5 else 1)
        assert cated[len(ds) + i][1] == (2 if i < 5 else 3)


def test_mixup():
    ds = StupidDataset()
    md = MixUpDataset(ds)
    md[0]


def test_cached():
    ds = StupidDataset()
    md = CachedDataset(ds)
    md[0]


def test_withindex():
    ds = StupidDataset()
    md = WithIndexDataset(ds)
    md[0]
