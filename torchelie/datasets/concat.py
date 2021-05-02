from torch.utils.data import Dataset
from torchelie.utils import indent
from typing import List, Tuple, Any, Generic, TypeVar, Sequence
from typing import overload

T = TypeVar('T')
U = TypeVar('U')


class CatedSamples(Generic[T]):
    def __init__(self, samples: List[List[Tuple[T, int]]]) -> None:
        self.samples = samples
        self.n_classes = [
            len(set(samp[1] for samp in sample)) for sample in samples
        ]

    def __len__(self) -> int:
        return sum(len(ds) for ds in self.samples)

    def __getitem__(self, i: int) -> Tuple[T, int]:
        class_offset = 0
        for samp, n_class in zip(self.samples, self.n_classes):
            if i < len(samp):
                return samp[i][0], samp[i][1] + class_offset
            i -= len(samp)
            class_offset += n_class
        raise IndexError


class CatedLists(Sequence[T]):
    def __init__(self, ls: List[List[T]]) -> None:
        self.ls = ls

    def __len__(self) -> int:
        return sum([len(ds) for ds in self.ls])

    @overload
    def __getitem__(self, i: int) -> T:
        ...

    @overload
    def __getitem__(self, i: slice) -> Sequence[T]:
        ...

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[ii] for ii in range(*i.indices(len(self)))]

        for catedlist in self.ls:
            if i < len(catedlist):
                return catedlist[i]
            i -= len(catedlist)
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
    classes: CatedLists[str]

    def __init__(self, datasets: List) -> None:
        self.datasets = datasets

        self.classes = CatedLists([ds.classes for ds in datasets])
        self.samples = CatedSamples([ds.samples for ds in datasets])
        self.class_to_idx = {nm: i for i, nm in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[Any, int]:
        class_offset = 0
        for ds in self.datasets:
            if i < len(ds):
                x, t = ds[i]
                return x, t + class_offset
            i -= len(ds)
            class_offset += len(ds.classes)
        raise IndexError

    def __repr__(self) -> str:
        return "DatasetConcat:\n" + '\n--\n'.join(
            [indent(repr(d)) for d in self.datasets])


class MergedSamples:
    def __init__(self, ds) -> None:
        self.ds = ds

    def __len__(self) -> int:
        return sum(len(d) for d in self.ds.datasets)

    def __getitem__(self, i: int):
        for ds in self.ds.datasets:
            if i < len(ds):
                x, y, *ys = ds.samples[i]
                return [x, self.ds.class_to_idx[ds.classes[y]]] + ys
            i -= len(ds)
        raise IndexError


class MergedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.classes = list(set(c for d in datasets for c in d.classes))
        self.classes.sort()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = MergedSamples(self)

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
        return "MergedDatasets: \n" + "\n".join(
            [indent(repr(ds)) for ds in self.datasets])
