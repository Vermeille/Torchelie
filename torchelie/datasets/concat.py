import torch


class CatedSamples:
    def __init__(self, samples):
        self.samples = samples
        self.n_classes = [
            len(set(samp[1] for samp in sample)) for sample in samples
        ]

    def __len__(self):
        return sum(len(ds) for ds in self.samples)

    def __getitem__(self, i):
        class_offset = 0
        for samp, n_class in zip(self.samples, self.n_classes):
            if i < len(samp):
                return samp[i][0], samp[i][1] + class_offset
            i -= len(samp)
            class_offset += n_class
        raise IndexError


class CatedLists:
    def __init__(self, ls):
        self.ls = ls

    def __len__(self):
        return sum([len(ds) for ds in self.ls])

    def __getitem__(self, i):
        for l in self.ls:
            if i < len(l):
                return l[i]
            i -= len(l)
        raise IndexError


class HorizontalConcatDataset(torch.utils.data.Dataset):
    """
    Concatenates multiple datasets. However, while torchvision's ConcatDataset
    just concatenates samples, torchelie's also relabels classes. While a
    vertical concat like torchvision's is useful to add more examples per
    class, an horizontal concat merges datasets to more classes.

    Args:
        datasets (list of Dataset): the datasets to concatenate
    """
    def __init__(self, datasets):
        self.datasets = datasets

        self.classes = CatedLists([ds.classes for ds in datasets])
        self.samples = CatedSamples([ds.samples for ds in datasets])
        self.class_to_idx = {nm: i for i, nm in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        class_offset = 0
        for ds in self.datasets:
            if i < len(ds):
                x, t = ds[i]
                return x, t + class_offset
            i -= len(ds)
            class_offset += len(ds.classes)
        raise IndexError

    def __repr__(self):
        return "DatasetConcat(" + ", ".join([repr(d)
                                             for d in self.datasets]) + ")"

class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.classes = list(set(c for d in datasets for c in d.classes))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        class MergedSamples:
            def __len__(self2):
                return sum(len(d) for d in self.datasets)

            def __getitem__(self2, i):
                for ds in self.datasets:
                    if i < len(ds):
                        x, y, *ys = ds.samples[i]
                        return [x, self.class_to_idx[ds.classes[y]]] + ys
                    i -= len(ds)
                raise IndexError

        self.samples = MergedSamples()

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, i):
        for ds in self.datasets:
            if i < len(ds):
                x, y, *ys = ds[i]
                return [x, self.class_to_idx[ds.classes[y]]] + ys
            i -= len(ds)
        raise IndexError

    def __repr__(self):
        return "MergedDatasets({})".format(self.datasets)
