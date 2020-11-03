import random
import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torchelie.utils as tu
import torchelie.nn as tnn
from .classifier import ConcatPoolClassifier1


class StyleGAN2Generator(nn.Module):
    """
    Generator from StyleGAN2

    Args:
        noise_size (int): size of the input gaussian noise
        ch_mul (float): width adaptor, multiplies the number of channels each
            layer (default: 1.)
        img_size (int): spatial size of the image to generate (default: 128)
        max_ch (int): maximum number of channels (default: 512)
        equal_lr (bool): whether to use equalized lr or not (default: True)
    """
    def __init__(self,
                 noise_size: int,
                 ch_mul: float = 1,
                 img_size: int = 128,
                 max_ch: int = 512,
                 equal_lr: bool = True):
        super().__init__()
        dyn = equal_lr
        self.encode = nn.Sequential(
            tu.kaiming(nn.Linear(noise_size, noise_size), dynamic=dyn, a=0.2),
            nn.LeakyReLU(0.2, True),
            tu.kaiming(nn.Linear(noise_size, noise_size), dynamic=dyn, a=0.2),
            nn.LeakyReLU(0.2, True),
            tu.kaiming(nn.Linear(noise_size, noise_size), dynamic=dyn, a=0.2),
            nn.LeakyReLU(0.2, True),
        )
        res = 4
        render = []
        while res <= img_size:
            ch = int(512 / (2**(math.log2(res) - 7)) * ch_mul)
            if res == 4:
                render.append(('const', tnn.Const(min(max_ch, ch), 4, 4)))
            render.append((f'conv{res}x{res}',
                           tnn.StyleGAN2Block(min(max_ch, ch),
                                              min(max_ch, ch // 2),
                                              noise_size,
                                              upsample=res != 4,
                                              n_layers=1 if res == 4 else 2,
                                              equal_lr=equal_lr)))
            res *= 2

        self.render = tnn.CondSeq(OrderedDict(render))

    def discretize_some(self, z):
        # This is just experimental
        discrete = min(z.shape[1] // 4, 32)
        z = torch.cat([
            z[:, :discrete].ge(0).float() + z[:, :discrete] - z[:,
                :discrete].detach(),
            z[:, discrete:].pow(2).mean(1, keepdim=True).rsqrt() *
            z[:, discrete:]
        ],
                      dim=1)
        return z

    def forward(self, z, mixing: bool = True) -> torch.Tensor:
        w1 = self.encode(self.discretize_some(z))
        if mixing:
            L = random.randint(0, len(self.render) - 1)
            self.render[:L].condition(w1)
            w2 = self.encode(self.discretize_some(torch.randn_like(z)))
            self.render[L:].condition(w2)
        else:
            self.render.condition(w1)
        return torch.sigmoid(self.render(w1)[0])

    def ppl(self, z, amp_scaler=None):
        with torch.cuda.amp.autocast(amp_scaler is not None):
            w = self.encode(self.discretize_some(z))
            self.render.condition(w)
            gen = torch.sigmoid(self.render(w)[0])
        B, C, H, W = gen.shape
        noise = torch.randn_like(gen) / math.sqrt(H * W)
        scaler = (lambda x: x) if amp_scaler is None else amp_scaler.scale
        JwTy = torch.autograd.grad(outputs=scaler(torch.sum(gen * noise)),
                                   inputs=w,
                                   create_graph=True,
                                   only_inputs=True)[0]
        JwTy = JwTy if amp_scaler is None else JwTy / amp_scaler.get_scale()
        JwTy_norm = JwTy.pow(2).sum(1).mul(1/(2*len(self.render))).sqrt()

        E_JwTy_norm = JwTy_norm.detach().mean().item()
        if hasattr(self, 'ppl_goal'):
            self.ppl_goal = 0.99 * self.ppl_goal + 0.01 * E_JwTy_norm
        else:
            self.ppl_goal = 0

        return (JwTy_norm - self.ppl_goal).pow(2).mean() * 2


def StyleGAN2Discriminator(input_sz,
                           max_ch: int = 512,
                           ch_mul: int = 1,
                           equal_lr: bool = True):
    """
    Build the discriminator for StyleGAN2

    Args:
        input_sz (int): image size
        max_ch (int): maximum number of channels (default: 512)
        ch_mul (float): multiply the number of channels on each layer by this
            value (default, 1.)
        equal_lr (bool): equalize the learning rates with dynamic weight
            scaling
    """
    dyn = equal_lr
    res = input_sz
    ch = int(512 / (2**(math.log2(res) - 6)) * ch_mul)
    layers = [(f'rgbto{res}x{res}',
               tu.xavier(tnn.Conv3x3(3, min(max_ch, ch)), dynamic=dyn))]

    while res > 4:
        res = res // 2
        layers.append((f'conv{res}x{res}',
                       tnn.ResidualDiscrBlock(min(max_ch, ch),
                                              min(max_ch, ch * 2),
                                              downsample=True,
                                              equal_lr=dyn)))
        ch *= 2
    layers.append(('relu1', nn.LeakyReLU(0.2, True)))
    layers.append(('mbstd', tnn.MinibatchStddev()))
    layers.append(
        (f'mbconv4x4',
         tu.kaiming(tnn.Conv3x3(min(max_ch, ch) + 1, min(max_ch, ch * 2)),
                    a=0.2,
                    dynamic=dyn)))
    layers.append(('relu2', nn.LeakyReLU(0.2, True)))
    model = nn.Sequential(OrderedDict(layers))
    model = ConcatPoolClassifier1(model, min(max_ch, ch), 1, 0.)
    tu.xavier(model.head[-1], dynamic=dyn)
    return model
