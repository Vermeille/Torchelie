import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from typing import Optional, Callable


class Pix2PixDataset(ImageFolder):
    """
    Paired images datasets made for the Pix2Pix paper for paired image
    translation.

    Args:
        root (str): path to the dataset
        which (str): which dataset to use. One of ['cityscapes',
            'edges2handbags', 'edges2shoes', 'facades', 'maps', 'night2day']
        download (bool): if True, download the dataset
        transform (optional: function): A callable that transforms the paired
            image represented as a 6 channels tensor.
    """
    def __init__(self,
                 root: str,
                 which: str,
                 download: bool = False,
                 transform: Optional[Callable] = None) -> None:
        assert which in [
            'cityscapes', 'edges2handbags', 'edges2shoes', 'facades', 'maps',
            'night2day'
        ]
        root = root + '/' + which
        if download:
            download_and_extract_archive(
                f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{which}.tar.gz',
                root)
        super().__init__(root, transform=transform)

    def _concat(self, x: torch.Tensor) -> torch.Tensor:
        w = x.shape[2] // 2
        return torch.cat([x[:, :, w:2 * w], x[:, :, :w]], dim=0)

    def _split(self, x):
        return x[:3], x[3:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the ith paired image as (img1, img2)
        """
        img_path, y = self.samples[i]
        img = to_tensor(Image.open(img_path))

        img = self._concat(img)

        if self.transform is not None:
            img = self.transform(img)
        return self._split(img)
