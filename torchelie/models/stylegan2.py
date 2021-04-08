import random
import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torchelie.utils as tu
import torchelie.nn as tnn
from .classifier import ConcatPoolClassifier1
from typing import List, Tuple


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
    w_avg: torch.Tensor

    def __init__(self,
                 noise_size: int,
                 ch_mul: float = 1,
                 img_size: int = 128,
                 max_ch: int = 512,
                 equal_lr: bool = True):
        super().__init__()
        dyn = equal_lr
        self.register_buffer('w_avg', torch.zeros(noise_size))

        def eq_lr_linear():
            if not equal_lr:
                return tu.kaiming(nn.Linear(noise_size, noise_size),
                                  a=0.2,
                                  dynamic=True)
            lr_mul = 0.01
            m = nn.Linear(noise_size, noise_size)
            m.weight.data.normal_(0, 1 / lr_mul)
            tnn.utils.weight_scale(m, scale=lr_mul / math.sqrt(noise_size))

            m.bias.data.normal_(0, 1 / lr_mul)
            tnn.utils.weight_scale(m, name='bias', scale=lr_mul)
            return m

        self.encode = nn.Sequential(
            eq_lr_linear(),
            nn.LeakyReLU(0.2, True),
            eq_lr_linear(),
            nn.LeakyReLU(0.2, True),
            eq_lr_linear(),
            nn.LeakyReLU(0.2, True),
        )
        res = 4
        render = tnn.ModuleGraph(f'rgb_{img_size}x{img_size}')

        while res <= img_size:
            ch = int(512 / (2**(math.log2(res) - 7)) * ch_mul)
            if res == 4:
                render.add_operation(inputs=['w'],
                                     outputs=[f'fmap_constxconst'],
                                     name='const',
                                     operation=tnn.Const(
                                         min(max_ch, ch), 4, 4))
                prev_res = 'const'
            else:
                prev_res = str(res // 2)

            block = tnn.StyleGAN2Block(min(max_ch, ch),
                                       min(max_ch, ch // 2),
                                       noise_size,
                                       upsample=res != 4,
                                       n_layers=1 if res == 4 else 2,
                                       equal_lr=equal_lr)
            render.add_operation(
                inputs=[
                    f'fmap_{prev_res}x{prev_res}', f'w_{res}x{res}',
                    f'rgb_{prev_res}x{prev_res}'
                ],
                operation=block,
                name=f'conv{res}x{res}',
                outputs=[f'rgb_{res}x{res}', f'fmap_{res}x{res}'])
            res *= 2

        print(render)
        self.render = render

    def discretize_some(self, z):
        # This is just experimental
        discrete = min(z.shape[1] // 4, 32)
        z = torch.cat([
            z[:, :discrete].ge(0).float() + z[:, :discrete] -
            z[:, :discrete].detach(),
            z[:, discrete:].pow(2).mean(1, keepdim=True).rsqrt() *
            z[:, discrete:]
        ],
                      dim=1)
        return z

    def forward(self, z, mixing: bool = True) -> torch.Tensor:
        w = self.encode(self.discretize_some(z))

        if self.training:
            self.w_avg = 0.995 * self.w_avg + (1 - 0.995) * w.detach().mean(0)

        L = random.randint(0, len(self.render) - 2)  # ignore the nn.Identity
        mix_res = 2**(L + 2)
        ws = {'w': w}
        for i_res in range(len(self.render) - 1):  # ignore the nn.Idendity
            res = 2**(i_res + 2)
            if mixing and res == mix_res and random.uniform(0, 1) < 0.9:
                w = self.encode(self.discretize_some(torch.randn_like(z)))
            if res <= 32 and not self.training:
                ws[f'w_{res}x{res}'] = 0.3 * self.w_avg + 0.7 * w
            else:
                ws[f'w_{res}x{res}'] = w
        return torch.sigmoid(self.render(rgb_constxconst=None, **ws))

    def w_to_dict(self, w):
        return {
            'w': w,
            **{
                f'w_{2**(i+2)}x{2**(i+2)}': w
                for i in range(len(self.render) - 1)
            }
        }

    def ppl(self, z):
        w = self.encode(self.discretize_some(z))
        gen = torch.sigmoid(
            self.render(rgb_constxconst=None, **self.w_to_dict(w))['out'])
        B, C, H, W = gen.shape
        noise = torch.randn_like(gen) / math.sqrt(H * W)
        JwTy = torch.autograd.grad(outputs=torch.sum(gen * noise),
                                   inputs=w,
                                   create_graph=True,
                                   only_inputs=True)[0]
        JwTy_norm = JwTy.pow(2).sum(1).mul(1 / (2 * len(self.render))).sqrt()

        E_JwTy_norm = JwTy_norm.detach().mean().item()
        if hasattr(self, 'ppl_goal'):
            self.ppl_goal = 0.99 * self.ppl_goal + 0.01 * E_JwTy_norm
        else:
            self.ppl_goal = 0

        return (JwTy_norm - self.ppl_goal).pow(2).mean() * 2


def StyleGAN2Discriminator(input_sz,
                           max_ch: int = 512,
                           ch_mul: int = 1,
                           equal_lr: bool = True,
                           input_ch: int = 3,
                           with_minibatch_std: bool = True):
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
    layers: List[Tuple[str, nn.Module]] = [
        (f'rgbto{res}x{res}',
         tu.xavier(tnn.Conv3x3(input_ch, min(max_ch, ch)),
                   dynamic=dyn,
                   mode='fan_in'))
    ]

    while res > 4:
        res = res // 2
        layers.append((f'conv{res}x{res}',
                       tnn.ResidualDiscrBlock(min(max_ch, ch),
                                              min(max_ch, ch * 2),
                                              downsample=True,
                                              equal_lr=dyn)))
        ch *= 2
    layers.append(('relu1', nn.LeakyReLU(0.2, True)))
    if with_minibatch_std:
        layers.append(('mbstd', tnn.MinibatchStddev()))
        layers.append(
            (f'mbconv4x4',
             tu.kaiming(tnn.Conv3x3(min(max_ch, ch) + 1, min(max_ch, ch * 2)),
                        a=0.2,
                        dynamic=dyn,
                        mode='fan_in')))
        layers.append(('relu2', nn.LeakyReLU(0.2, True)))
    model: nn.Module = nn.Sequential(OrderedDict(layers))
    model = ConcatPoolClassifier1(model, min(max_ch, ch), 1, 0.)
    tu.xavier(model.head[-1], dynamic=dyn, mode='fan_in')
    return model
