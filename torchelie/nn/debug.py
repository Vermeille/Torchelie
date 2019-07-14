import torch.nn as nn

class Debug(nn.Module):
    def __init__(self, name):
        super(Debug, self).__init__()
        self.name = name

    def forward(self, x):
        print("{} mean {:.2f} {:.2f} var s{:.2f} {:.2f}".format(
            self.name,
            x.mean(dim=[0, 2, 3]).mean().item(),
            x.mean().item(),
            x.std(dim=[0, 2, 3]).mean().item(),
            x.std().item()))
        return x

