import os
from typing import Optional, Callable
from typing_extensions import Literal
import torch
import torchvision.transforms as TF
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from torchvision.datasets.utils import download_and_extract_archive

__all__ = ['ColoredColumns', 'ColoredRows', 'Imagenette', 'Imagewoof']


class ColoredColumns(Dataset):
    """
    A dataset of precedurally generated images of columns randomly colorized.

    Args:
        *size (int): size of images
        transform (transforms or None): the image transforms to apply to the
            generated pictures
    """

    def __init__(self, *size, transform=None) -> None:
        super(ColoredColumns, self).__init__()
        self.size = size
        self.transform = transform if transform is not None else (lambda x: x)

    def __len__(self):
        return 10000

    def __getitem__(self, i):
        cols = torch.randint(0, 255, (3, 1, self.size[1]))
        expanded = cols.expand(3, *self.size).float()
        img = TF.ToPILImage()(expanded / 255)
        return self.transform(img), 0


class ColoredRows(Dataset):
    """
    A dataset of precedurally generated images of rows randomly colorized.

    Args:
        *size (int): size of images
        transform (transforms or None): the image transforms to apply to the
            generated pictures
    """

    def __init__(self, *size, transform=None) -> None:
        super(ColoredRows, self).__init__()
        self.size = size
        self.transform = transform if transform is not None else (lambda x: x)

    def __len__(self):
        return 10000

    def __getitem__(self, i):
        rows = torch.randint(0, 255, (3, self.size[0], 1))
        expanded = rows.expand(3, *self.size).float()
        img = TF.ToPILImage()(expanded / 255)
        return self.transform(img), 0


class Imagenette(ImageFolder):
    """
    Imagenette by Jeremy Howards ( https://github.com/fastai/imagenette ).

    Args:
        root (str): root directory
        split (bool): if False, use validation split
        transform (Callable): image transforms
        download (bool): if True and root empty, download the dataset
        version (str): which resolution to download ('full', '32Opx', '160px')

    """

    def __init__(self,
                 root: str,
                 train: bool,
                 transform: Optional[Callable] = None,
                 download: bool = False,
                 version: Literal['full', '320px', '160px'] = '320px'):
        size = ({
            'full': 'imagenette2',
            '320px': 'imagenette2-320',
            '160px': 'imagenette2-160'
        })[version]

        split = 'train' if train else 'val'
        if not self._check_integrity(f'{root}/{size}') and download:
            download_and_extract_archive(
                f'https://s3.amazonaws.com/fast-ai-imageclas/{size}.tgz',
                root,
                remove_finished=True)

        super().__init__(f'{root}/{size}/{split}', transform=transform)

    def _check_integrity(self, path):
        return os.path.exists(os.path.expanduser(path))


class Imagewoof(ImageFolder):
    """
    Imagewoof by Jeremy Howards ( https://github.com/fastai/imagenette ).

    Args:
        root (str): root directory
        split (bool): if False, use validation split
        transform (Callable): image transforms
        download (bool): if True and root empty, download the dataset
        version (str): which resolution to download ('full', '32Opx', '160px')

    """

    def __init__(self,
                 root: str,
                 train: bool,
                 transform: Optional[Callable] = None,
                 download: bool = False,
                 version: Literal['full', '320px', '160px'] = '320px'):
        size = ({
            'full': 'imagewoof2',
            '320px': 'imagewoof2-320',
            '160px': 'imagewoof2-160'
        })[version]

        split = 'train' if train else 'val'
        if not self._check_integrity(f'{root}/{size}') and download:
            download_and_extract_archive(
                f'https://s3.amazonaws.com/fast-ai-imageclas/{size}.tgz',
                root,
                remove_finished=True)

        super().__init__(f'{root}/{size}/{split}', transform=transform)

    def _check_integrity(self, path):
        return os.path.exists(os.path.expanduser(path))
