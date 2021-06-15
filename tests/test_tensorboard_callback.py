import pytest
import torch
import torchelie.callbacks as tcb
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import PILToTensor


@pytest.mark.require_tensorboard
def test_tesorboard():
    from torchelie.recipes import Recipe

    batch_size = 4

    class Dataset:
        def __init__(self, batch_size):
            self.batch_size = batch_size
            self.mnist = FashionMNIST('.', download=True, transform=PILToTensor())
            self.classes = self.mnist.classes
            self.num_classes = len(self.mnist.class_to_idx)
            self.target_by_classes = [[idx for idx in range(len(self.mnist)) if self.mnist.targets[idx] == i]
                                      for i in range(self.num_classes)]

        def __len__(self):
            return self.batch_size * self.num_classes

        def __getitem__(self, item):
            idx = self.target_by_classes[item//self.batch_size][item]
            x, y = self.mnist[idx]
            x = torch.stack(3*[x]).squeeze()
            x[2] = 0
            return x, y

    dst = Dataset(batch_size)

    def train(b):
        x, y = b
        return {'letter_number_int':     int(y[0]),
                'letter_number_tensor':  y[0],
                'letter_text':  dst.classes[int(y[0])],
                'test_html':  '<b>test HTML</b>',
                'letter_gray_img_HW':   x[0, 0],
                'letter_gray_img_CHW':   x[0, :1],
                'letter_gray_imgs_NCHW':  x[:, :1],
                'letter_color_img_CHW':  x[0],
                'letter_color_imgs_NCHW': x}

    r = Recipe(train, DataLoader(dst, batch_size))
    r.callbacks.add_callbacks([
        tcb.Counter(),
        tcb.TensorboardLogger(log_every=1),
        tcb.Log('letter_number_int', 'letter_number_int'),
        tcb.Log('letter_number_tensor', 'letter_number_tensor'),
        tcb.Log('letter_text', 'letter_text'),
        tcb.Log('test_html', 'test_html'),
        tcb.Log('letter_gray_img_HW', 'letter_gray_img_HW'),
        tcb.Log('letter_gray_img_CHW', 'letter_gray_img_CHW'),
        tcb.Log('letter_gray_imgs_NCHW', 'letter_gray_imgs_NCHW'),
        tcb.Log('letter_color_img_CHW', 'letter_color_img_CHW'),
        tcb.Log('letter_color_imgs_NCHW', 'letter_color_imgs_NCHW'),
    ])
    r.run(1)


if __name__ == '__main__':
    test_tesorboard()
