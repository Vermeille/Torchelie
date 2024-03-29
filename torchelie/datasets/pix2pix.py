import os
from typing import Optional, Callable, Tuple, List

import torch
from torchvision.transforms.functional import to_tensor
from torchvision.datasets.utils import download_and_extract_archive
from PIL import Image

from torchelie.utils import indent


class UnlabeledImages:
    """
    Serve all the images contained in a directory and subdirectories without
    any labels and structure constraint.

    Args:
        root (str): path to the root directory
        transform (callable): transformations
    """
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None) -> None:
        root = os.path.expanduser(root)
        self.root = root
        self.samples = list(
            root + '/' + name for root, _, files in os.walk(root)
            for name in files
            if name.split('.')[-1].lower() in ['bmp', 'jpg', 'jpeg', 'png'])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        img = Image.open(self.samples[i]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return [img]

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}:\n'
                f'    num_samples: {len(self)}\n'
                f'    root: {self.root}\n'
                f'    transform:\n'
                f'{indent(repr(self.transform), 8)}\n')


class ImagesPaths:
    """
    Serve all the images given in :code:`paths`.

    Args:
        paths (List[str]): paths to images
        transform (callable): transformations
    """
    def __init__(self,
                 paths: List[str],
                 transform: Optional[Callable] = None) -> None:
        self.samples = list(map(os.path.expanduser, paths))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        img = Image.open(self.samples[i])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}:\n'
                f'    num_samples: {len(self)}\n'
                f'    transform:\n'
                f'{indent(repr(self.transform), 8)}')


class SideBySideImagePairsDataset(UnlabeledImages):
    """
    Dataset for side-by-side images. It splits the images so that the same
    transforms are applied to pairs and remain meaningful.
    """
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 pre_split_transform: Optional[Callable] = None) -> None:
        self.pre_split_transform = pre_split_transform
        super().__init__(root, transform)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the ith paired image as (img1, img2)
        """
        img_path = self.samples[i]
        img = Image.open(img_path)
        if self.pre_split_transform is not None:
            img = self.pre_split_transform(img)
        img = to_tensor(img)

        img = self._concat(img)

        if self.transform is not None:
            img = self.transform(img)
        return self._split(img)

    def _concat(self, x: torch.Tensor) -> torch.Tensor:
        w = x.shape[2] // 2
        return torch.cat([x[:, :, w:2 * w], x[:, :, :w]], dim=0)

    def _split(self, x):
        c = x.shape[0] // 2
        return x[:c], x[c:]


class Pix2PixDataset(SideBySideImagePairsDataset):
    """
    Paired images datasets made for the Pix2Pix paper for paired image
    translation.

    Args:
        root (str): path to the dataset
        which (str): which dataset to use. One of ['cityscapes',
            'edges2handbags', 'edges2shoes', 'facades', 'maps', 'night2day']
        split (str): choose 'train', 'val' or 'test' set. Default to train.
        download (bool): if True, download the dataset
        transform (optional: function): A callable that transforms the paired
            image represented as a 6 channels tensor.


    cityscapes, maps have train and val splits.

    night2day, facades have traint, test and val splits.
    """
    def __init__(self,
                 root: str,
                 which: str,
                 split: str = 'train',
                 download: bool = False,
                 transform: Optional[Callable] = None) -> None:
        assert which in [
            'cityscapes', 'edges2handbags', 'edges2shoes', 'facades', 'maps',
            'night2day'
        ], f'{which} is not a valid dataset for pix2pix'
        root = root
        if not self._check_integrity(f'{root}/{which}') and download:
            download_and_extract_archive(
                f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{which}.tar.gz',
                root,
                remove_finished=True)
        super().__init__(f'{root}/{which}/{split}', transform=transform)

    def _check_integrity(self, path):
        return os.path.exists(os.path.expanduser(path))
