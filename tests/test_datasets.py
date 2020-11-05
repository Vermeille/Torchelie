from torchelie.datasets import *


class StupidDataset:
    def __init__(self):
        """
        Initialize the classes

        Args:
            self: (todo): write your description
        """
        self.classes = [0, 1]
        self.imgs = [(i, 0 if i < 5 else 1) for i in range(10)]
        self.samples = self.imgs

    def __len__(self):
        """
        Returns the number of bytes in bytes.

        Args:
            self: (todo): write your description
        """
        return 10

    def __getitem__(self, i):
        """
        Return the item at the given index.

        Args:
            self: (todo): write your description
            i: (todo): write your description
        """
        return torch.FloatTensor([i]), torch.LongTensor([self.imgs[i][1]])


def test_colored():
    """
    Test if a column is an explanation

    Args:
    """
    cc = ColoredColumns(64, 64)
    cr = ColoredRows(64, 64)

    cc[0]
    cr[0]


def test_paired():
    """
    Test for all of the dataset.

    Args:
    """
    ds = StupidDataset()
    ds2 = StupidDataset()

    pd = PairedDataset(ds, ds2)
    assert len(pd) == 100
    assert pd[0] == list(zip(ds[0], ds2[0]))


def test_cat():
    """
    Test the dataset * dataset

    Args:
    """
    ds = StupidDataset()
    ds2 = StupidDataset()

    cated = HorizontalConcatDataset([ds, ds2])
    assert len(cated) == len(ds) * 2
    print([cated[i][1] for i in range(20)])

    for i in range(len(ds)):
        assert cated[i][1] == (0 if i < 5 else 1)
        assert cated[len(ds) + i][1] == (2 if i < 5 else 3)


def test_mixup():
    """
    The test test test test.

    Args:
    """
    ds = StupidDataset()
    md = MixUpDataset(ds)
    md[0]


def test_cached():
    """
    Returns the test test dataset.

    Args:
    """
    ds = StupidDataset()
    md = CachedDataset(ds)
    md[0]


def test_withindex():
    """
    Returns a test index of the dataset.

    Args:
    """
    ds = StupidDataset()
    md = WithIndexDataset(ds)
    md[0]
