import torch
import torchvision.transforms as TF


class ColoredColumns(torch.utils.data.Dataset):
    def __init__(self, *size, transform=None):
        super(ColoredColumns, self).__init__()
        self.size = size
        self.transform = transform if transform is not None else (lambda x:x)

    def __len__(self):
        return 10000

    def __getitem__(self, i):
        cols = torch.randint(0, 255, (3, 1, self.size[1]))
        expanded = cols.expand(3, *self.size).float()
        img = TF.ToPILImage()(expanded / 255)
        return self.transform(img), 0


class ColoredRows(torch.utils.data.Dataset):
    def __init__(self, *size, transform=None):
        super(ColoredRows, self).__init__()
        self.size = size
        self.transform = transform if transform is not None else (lambda x:x)

    def __len__(self):
        return 10000

    def __getitem__(self, i):
        rows = torch.randint(0, 255, (3, self.size[0], 1))
        expanded = rows.expand(3, *self.size).float()
        img = TF.ToPILImage()(expanded / 255)
        return self.transform(img), 0

