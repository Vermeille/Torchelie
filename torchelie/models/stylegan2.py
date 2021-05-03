import random
import math
import torch
import torch.nn as nn
import torchelie.utils as tu
import torchelie.nn as tnn
from typing import List, Tuple, Union, overload
from typing_extensions import Literal
from .classifier import ClassificationHead
from .snres_discr import ResidualDiscriminator
from torchelie.transforms.differentiable import BinomialFilter2d


class LinearReLU(tnn.CondSeq):
    relu: nn.Module
    linear: nn.Module

    @tu.experimental
    def __init__(self, in_features: int, out_features) -> None:
        super().__init__()
        self.linear = tu.kaiming(nn.Linear(in_features, out_features))
        self.relu = nn.ReLU(True)

    def leaky(self) -> 'LinearReLU':
        if isinstance(self.relu, (nn.ReLU, nn.LeakyReLU)):
            self.relu = nn.LeakyReLU(0.2, self.relu.inplace)
        else:
            self.relu = nn.LeakyReLU(0.2, True)
        return self

    def to_equal_lr(self) -> 'LinearReLU':
        if isinstance(self.relu, nn.LeakyReLU):
            tu.kaiming(self.linear, dynamic=True, a=self.relu.negative_slope)
        else:
            tu.kaiming(self.linear, dynamic=True)
        return self

    def to_differential_lr(self, lr: float):
        tnn.utils.weight_scale(self.linear, name='bias', scale=lr)
        if isinstance(self.relu, nn.LeakyReLU):
            g = tu.kaiming_gain(self.linear, a=self.relu.negative_slope)
        else:
            g = tu.kaiming_gain(self.linear)
        tnn.utils.weight_scale(self.linear, scale=lr * g)
        assert isinstance(self.linear.weight_g, torch.Tensor)
        self.linear.weight_g.data.normal_(0, 1. / lr)
        return self


class MappingNetwork(tnn.CondSeq):
    def __init__(self, num_features: int, num_layers: int = 3) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(f'linear_{i}',
                            LinearReLU(num_features, num_features))

    def leaky(self) -> 'MappingNetwork':
        for m in self:
            m.leaky()
        return self

    def to_equal_lr(self) -> 'MappingNetwork':
        for m in self:
            m.to_equal_lr()
        return self

    def to_differential_lr(self, lr: float) -> 'MappingNetwork':
        for m in self:
            m.to_differential_lr(lr)
        return self


class StyleGAN2Generator(nn.Module):
    """
    Generator from StyleGAN2

    Args:
        noise_size (int): size of the input gaussian noise
        ch_mul (float): width adaptor, multiplies the number of channels each
            layer (default: 1.)
        img_size (int): spatial size of the image to generate (default: 128)
        max_ch (int): maximum number of channels (default: 512)
    """
    w_avg: torch.Tensor

    @tu.experimental
    def __init__(self,
                 noise_size: int,
                 ch_mul: float = 1,
                 img_size: int = 128,
                 max_ch: int = 512):
        super().__init__()
        self.register_buffer('w_avg', torch.zeros(noise_size))

        self.encode = MappingNetwork(noise_size, 3).to_differential_lr(0.01)
        self.encode.leaky()
        res = 4
        render = tnn.ModuleGraph('out')

        while res <= img_size:
            ch = int(512 / (2**(math.log2(res) - 7)) * ch_mul)
            if res == 4:
                render.add_operation(inputs=['N'],
                                     outputs=['fmap_constxconst'],
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
                                       n_layers=1 if res == 4 else 2)
            render.add_operation(
                inputs=[
                    f'fmap_{prev_res}x{prev_res}', f'w_{res}x{res}',
                    f'rgb_{prev_res}x{prev_res}'
                ],
                operation=block,
                name=f'conv{res}x{res}',
                outputs=[f'rgb_{res}x{res}', f'fmap_{res}x{res}'])
            res *= 2

        render.add_operation(inputs=[f'rgb_{img_size}x{img_size}'],
                             operation=nn.Sigmoid(),
                             name='sigmoid',
                             outputs=['out'])

        self.render = render

    @overload
    def forward(self,
                z,
                mixing: bool,
                get_w: Literal[False]) -> torch.Tensor:
        ...

    @overload
    def forward(
            self,
            z,
            mixing: bool,
            get_w: Literal[True]) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def forward(self, z, mixing: bool = True, get_w: bool = False):
        w = self.encode(z)

        if self.training:
            self.w_avg.copy_(tu.lerp(w.detach().mean(0), self.w_avg, 0.995))

        L = random.randint(0, len(self.render) - 1)
        mix_res = 2**(L + 2)
        ws = {'N': w.shape[0]}
        for i_res in range(len(self.render)):
            res = 2**(i_res + 2)
            if mixing and res == mix_res and random.uniform(0, 1) < 0.9:
                w = self.encode(torch.randn_like(z))

            if res <= 32 and not self.training:
                ws[f'w_{res}x{res}'] = 0.3 * self.w_avg + 0.7 * w
            else:
                ws[f'w_{res}x{res}'] = w

        if not get_w:
            return self.render(rgb_constxconst=None, **ws)
        else:
            return self.render(rgb_constxconst=None, **ws), w

    def w_to_dict(self, w):
        return {
            'N': w.shape[0],
            **{
                f'w_{2**(i+2)}x{2**(i+2)}': w
                for i in range(len(self.render) - 1)
            }
        }


@tu.experimental
def StyleGAN2Discriminator(input_sz,
                           max_ch: int = 512,
                           ch_mul: int = 1) -> ResidualDiscriminator:
    """
        Build the discriminator for StyleGAN2

        Args:
            input_sz (int): image size
            max_ch (int): maximum number of channels (default: 512)
            ch_mul (float): multiply the number of channels on each layer by this
                value (default, 1.)
    """
    import math
    res = input_sz
    ch = int(512 / (2**(math.log2(res) - 6)) * ch_mul)
    layers: List[Union[str, int]] = [min(max_ch, ch)]

    while res > 4:
        res = res // 2
        layers.append('D')
        layers.append(min(max_ch, ch * 2))
        ch *= 2

    net = ResidualDiscriminator(layers)

    def make_binomial(m):
        if isinstance(m, nn.AvgPool2d):
            return BinomialFilter2d(2)
        return m

    tnn.utils.edit_model(net, make_binomial)
    net.add_minibatch_stddev()
    assert isinstance(net.classifier, ClassificationHead)
    net.classifier.to_two_layers(min(ch, max_ch))
    net.classifier.set_pool_size(4)
    net.classifier.leaky()
    net.to_equal_lr()
    return net
