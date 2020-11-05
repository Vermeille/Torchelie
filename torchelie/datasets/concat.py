import torch
from torch.utils.data import Dataset


class CatedSamples:
    def __init__(self, samples):
        """
        Initialize samples.

        Args:
            self: (todo): write your description
            samples: (list): write your description
        """
        self.samples = samples
        self.n_classes = [
            len(set(samp[1] for samp in sample)) for sample in samples
        ]

    def __len__(self):
        """
        Returns the length of the samples.

        Args:
            self: (todo): write your description
        """
        return sum(len(ds) for ds in self.samples)

    def __getitem__(self, i):
        """
        Return the item at i.

        Args:
            self: (todo): write your description
            i: (todo): write your description
        """
        class_offset = 0
        for samp, n_class in zip(self.samples, self.n_classes):
            if i < len(samp):
                return samp[i][0], samp[i][1] + class_offset
            i -= len(samp)
            class_offset += n_class
        raise IndexError


class CatedLists:
    def __init__(self, ls):
        """
        Initialize the module

        Args:
            self: (todo): write your description
            ls: (array): write your description
        """
        self.ls = ls

    def __len__(self):
        """
        Return the number of rows in - place

        Args:
            self: (todo): write your description
        """
        return sum([len(ds) for ds in self.ls])

    def __getitem__(self, i):
        """
        Returns the item from the list.

        Args:
            self: (todo): write your description
            i: (todo): write your description
        """
        for l in self.ls:
            if i < len(l):
                return l[i]
            i -= len(l)
        raise IndexError


class HorizontalConcatDataset(Dataset):
    """
    Concatenates multiple datasets. However, while torchvision's ConcatDataset
    just concatenates samples, torchelie's also relabels classes. While a
    vertical concat like torchvision's is useful to add more examples per
    class, an horizontal concat merges datasets to more classes.

    Args:
        datasets (list of Dataset): the datasets to concatenate
    """
    def __init__(self, datasets):
        """
        Initialize the classes.

        Args:
            self: (todo): write your description
            datasets: (todo): write your description
        """
        self.datasets = datasets

        self.classes = CatedLists([ds.classes for ds in datasets])
        self.samples = CatedSamples([ds.samples for ds in datasets])
        self.class_to_idx = {nm: i for i, nm in enumerate(self.classes)}

    def __len__(self):
        """
        Returns the number of samples.

        Args:
            self: (todo): write your description
        """
        return len(self.samples)

    def __getitem__(self, i):
        """
        Return the item at the dataset.

        Args:
            self: (todo): write your description
            i: (todo): write your description
        """
        class_offset = 0
        for ds in self.datasets:
            if i < len(ds):
                x, t = ds[i]
                return x, t + class_offset
            i -= len(ds)
            class_offset += len(ds.classes)
        raise IndexError

    def __repr__(self):
        """
        Return a representation of the representation of this field.

        Args:
            self: (todo): write your description
        """
        return "DatasetConcat(" + ", ".join([repr(d)
                                             for d in self.datasets]) + ")"

class MergedSamples:
    def __init__(self, ds):
        """
        Initialize the dataset.

        Args:
            self: (todo): write your description
            ds: (todo): write your description
        """
        self.ds = ds

    def __len__(self):
        """
        Returns the number of datasets.

        Args:
            self: (todo): write your description
        """
        return sum(len(d) for d in self.ds.datasets)

    def __getitem__(self, i):
        """
        Returns the dataset from the given index.

        Args:
            self: (todo): write your description
            i: (todo): write your description
        """
        for ds in self.ds.datasets:
            if i < len(ds):
                x, y, *ys = ds.samples[i]
                return [x, self.ds.class_to_idx[ds.classes[y]]] + ys
            i -= len(ds)
        raise IndexError


class MergedDataset(Dataset):
    def __init__(self, datasets):
        """
        Initialize the classes.

        Args:
            self: (todo): write your description
            datasets: (todo): write your description
        """
        self.datasets = datasets
        self.classes = list(set(c for d in datasets for c in d.classes))
        self.classes.sort()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}


        self.samples = MergedSamples(self)

    def __len__(self):
        """
        The sum of the datasets.

        Args:
            self: (todo): write your description
        """
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, i):
        """
        Return the dataset corresponding to i

        Args:
            self: (todo): write your description
            i: (todo): write your description
        """
        for ds in self.datasets:
            if i < len(ds):
                x, y, *ys = ds[i]
                return [x, self.class_to_idx[ds.classes[y]]] + ys
            i -= len(ds)
        raise IndexError

    def __repr__(self):
        """
        Return a human - readable representation of this object.

        Args:
            self: (todo): write your description
        """
        return "MergedDatasets({})".format(self.datasets)
