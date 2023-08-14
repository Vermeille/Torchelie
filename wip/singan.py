import torch
import torch.nn as nn
import torchelie.nn as tnn
import torchelie.utils as tu
import torch.nn.functional as F


class SinGANGenerator(tnn.CondSeq):
    @tu.experimental
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.add_module(
            'conv1',
            tnn.ConvBlock(in_channels, hidden_channels, 3).leaky())
        self.add_module(
            'conv2',
            tnn.ConvBlock(hidden_channels, hidden_channels, 3).leaky())
        self.add_module(
            'conv3',
            tnn.ConvBlock(hidden_channels, hidden_channels, 3).leaky())
        self.add_module(
            'conv4',
            tnn.ConvBlock(hidden_channels, hidden_channels, 3).leaky())
        self.add_module('conv5', tnn.ConvBlock(hidden_channels, 3, 3))
        del self.conv5.relu

    def dynamic_init(self) -> None:
        for conv in self.modules():
            if isinstance(conv, nn.Conv2d):
                tu.kaiming(conv, mode='fan_in', dynamic=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(super().forward(x * 2 - 1 + torch.randn_like(x)))


class SinGANDiscriminator(tnn.CondSeq):
    @tu.experimental
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.add_module(
            'conv1',
            tnn.ConvBlock(in_channels, hidden_channels,
                             3).remove_batchnorm().leaky())
        self.add_module(
            'conv2',
            tnn.ConvBlock(hidden_channels, hidden_channels, 3).leaky())
        self.add_module(
            'conv3',
            tnn.ConvBlock(hidden_channels, hidden_channels, 3).leaky())
        self.add_module(
            'conv4',
            tnn.ConvBlock(hidden_channels, hidden_channels, 3).leaky())
        self.add_module(
            'conv5',
            tnn.ConvBlock(hidden_channels, 1, 3).remove_batchnorm())
        del self.conv5.relu

    def dynamic_init(self) -> None:
        for conv in self.modules():
            if isinstance(conv, nn.Conv2d):
                tu.kaiming(conv,
                           dynamic=True,
                           nonlinearity='leaky_relu',
                           mode='fan_in',
                           a=0.2)
        for conv in self.modules():
            if isinstance(conv, tnn.ConvBlock):
                conv.remove_batchnorm()

    #def forward(self, x: torch.Tensor) -> torch.Tensor: return super().forward(x * 2 - 1)


from typing import Optional, List, Any
from PIL import Image


class ImageList:
    @tu.experimental
    def __init__(self, images: List[str], transform: Optional[Any]) -> None:
        self.samples = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        img = Image.open(self.samples[i])
        if self.transform is not None:
            img = self.transform(img)
        return img


if __name__ == '__main__':
    from torchelie.recipes.gan import GANRecipe
    import torchelie.callbacks as tcb
    import torchelie.loss.gan.standard as gan_loss
    from torchelie.optim import RAdamW, Lookahead
    import torchvision as tv
    import torchvision.transforms as TF
    from torchelie.loss.gan.penalty import zero_gp
    G = SinGANGenerator(3, 128)
    G.dynamic_init()
    D = SinGANDiscriminator(3, 128)
    D.dynamic_init()
    print(G)
    print(D)

    ds = ImageList(['face.jpg'], TF.Compose([TF.Resize(16), TF.ToTensor()]))

    def G_fun(img):
        img = img.expand(10, -1, -1, -1)
        out = G(torch.zeros_like(img) + 0.5)
        pred = D(out * 2 - 1)
        loss = gan_loss.generated(pred)
        loss.backward()
        return {
            'G_loss': loss.item(),
            'out': out.detach().clamp(min=0, max=1),
            "ref": img.detach()
        }

    def D_fun(img):
        img = img.expand(10, -1, -1, -1)
        with torch.no_grad():
            out = G(torch.zeros_like(img) + 0.5)
        pred = D(out * 2 - 1)
        print(torch.sigmoid(pred))
        fake_loss = gan_loss.fake(pred)
        fake_loss.backward()

        pred = D(img * 2 - 1)
        real_loss = gan_loss.real(pred)
        real_loss.backward()

        gp, g_norm = zero_gp(D, img * 2 - 1, out * 2 - 1)
        (0.01 * gp).backward()
        return {'fake_loss': fake_loss.item(), 'Grad_norm': g_norm}

    def test_G(_):
        return {}

    recipe = GANRecipe(G,
                       D,
                       G_fun,
                       D_fun,
                       test_G,
                       ds,
                       visdom_env='singan',
                       test_every=50)
    recipe.callbacks.add_callbacks([
        tcb.Optimizer(
            Lookahead(
                RAdamW(D.parameters(),
                       lr=2e-3,
                       betas=(0., 0.99),
                       weight_decay=0))),
        tcb.WindowedMetricAvg('fake_loss'),
        tcb.Log('Grad_norm', 'grad_norm'),
    ])
    recipe.G_loop.callbacks.add_callbacks([
        tcb.WindowedMetricAvg('G_loss'),
        tcb.Log('out', 'out'),
        tcb.Log('ref', 'ref'),
        tcb.Optimizer(
            Lookahead(
                RAdamW(G.parameters(),
                       lr=4e-3,
                       betas=(0., 0.99),
                       weight_decay=0))),
    ])
    recipe.cuda()
    recipe.run(5000)
