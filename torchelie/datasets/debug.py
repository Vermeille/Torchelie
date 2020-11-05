import torch
import torchvision.transforms as TF
from torch.utils.data import Dataset


class ColoredColumns(Dataset):
    """
    A dataset of precedurally generated images of columns randomly colorized.

    Args:
        *size (int): size of images
        transform (transforms or None): the image transforms to apply to the
            generated pictures
    """
    def __init__(self, *size, transform=None):
        """
        Initialize the transform.

        Args:
            self: (todo): write your description
            size: (int): write your description
            transform: (str): write your description
        """
        super(ColoredColumns, self).__init__()
        self.size = size
        self.transform = transform if transform is not None else (lambda x:x)

    def __len__(self):
        """
        Returns the number of bytes in bytes

        Args:
            self: (todo): write your description
        """
        return 10000

    def __getitem__(self, i):
        """
        Get a pil image

        Args:
            self: (todo): write your description
            i: (todo): write your description
        """
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
    def __init__(self, *size, transform=None):
        """
        Initialize the transform.

        Args:
            self: (todo): write your description
            size: (int): write your description
            transform: (str): write your description
        """
        super(ColoredRows, self).__init__()
        self.size = size
        self.transform = transform if transform is not None else (lambda x:x)

    def __len__(self):
        """
        Returns the number of bytes in bytes

        Args:
            self: (todo): write your description
        """
        return 10000

    def __getitem__(self, i):
        """
        Return an image

        Args:
            self: (todo): write your description
            i: (todo): write your description
        """
        rows = torch.randint(0, 255, (3, self.size[0], 1))
        expanded = rows.expand(3, *self.size).float()
        img = TF.ToPILImage()(expanded / 255)
        return self.transform(img), 0

