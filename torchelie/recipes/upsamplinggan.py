from collections import OrderedDict
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchelie as tch
import torchelie.utils as tu
import torchelie.models.unet as U
from torchelie.recipes.gan import GANRecipe
from torchelie.transforms.differentiable import center_crop
import torchvision.transforms as TF
from PIL import Image

@tu.experimental
def jpeg_compress(x):
    import io
    with io.BytesIO() as f:
        x.save(f, format='jpeg', quality=random.randrange(10, 50),
                subsampling=0)
        f.seek(0)
        im = Image.open(f)
        im.load()
        return im

class ADATF:
    @tu.experimental
    def __init__(self, target_loss: float, growth: float = 0.01):
        self.p = 0
        self.target_loss = target_loss
        self.growth = growth

    def __call__(self, x, y):
        if self.p == 0:
            return x, y
        p = self.p
        RND = (x.shape[0], 1, 1, 1)
        x = torch.where(
            torch.rand(RND, device=x.device) < p / 2,
            tch.transforms.differentiable.roll(x, random.randint(-16, 16),
                                               random.randint(-16, 16)), x)
        color = tch.transforms.differentiable.AllAtOnceColor(x.shape[0])
        color.brightness(0.5, p)
        color.contrast(0.7, p)
        x = color.apply(x)
        y = color.apply(y)

        geom = tch.transforms.differentiable.AllAtOnceGeometric(x.shape[0])
        geom.translate(0.25, 0.25, p)
        geom.rotate(180, p)
        geom.scale(0.5, 0.5, p)
        geom.flip_x(0.5, p)
        geom.flip_y(0.5, p)
        x = geom.apply(x)
        y = geom.apply(y)
        return x, y

    def log_loss(self, l):
        if l > self.target_loss:
            self.p -= self.growth
        else:
            self.p += self.growth
        self.p = max(0, min(self.p, 0.9))

LEAK = 0.2
PAD_MODE = 'zeros'


class UBlock(nn.Module):
    @tu.experimental
    def __init__(self,
                 in_ch,
                 out_ch,
                 inner=None,
                 skip=True,
                 up_mode='bilinear'):
        super(UBlock, self).__init__()
        self.up_mode = up_mode
        self.encode = nn.Sequential(*[
            tch.nn.ResBlock(in_ch, out_ch if i == 0 else in_ch, norm=None)
            for i in range(2)
            ][::-1])

        self.inner = inner
        if inner is not None:
            self.skip = nn.Identity()

        inner_ch = out_ch * (1 if inner is None else 2)
        self.decode = nn.Sequential(*[
            tch.nn.Conv3x3(inner_ch, in_ch),
            tch.nn.Noise(in_ch),
            nn.LeakyReLU(0.2, True),
            tch.nn.Conv3x3(in_ch, in_ch),
            nn.LeakyReLU(0.2, True),
        ])

        self.to_rgb = nn.Sequential(
            OrderedDict([
                ('conv', tu.kaiming(nn.Conv2d(in_ch, 3, 3, padding=1)))
            ])
        )

    def forward(self, x_orig):
        x = self.encode(x_orig)
        if self.inner is not None:
            skip = x
            out = self.inner(F.avg_pool2d(x, 3, 2, 1))
            if isinstance(out, tuple):
                x, rgb = out
            else:
                x = out
                rgb = torch.zeros(x.shape[0],
                                  3,
                                  x.shape[2],
                                  x.shape[3],
                                  device=x.device)

            x = F.interpolate(x, mode=self.up_mode, size=skip.shape[2:])
            rgb = F.interpolate(rgb, mode=self.up_mode, size=skip.shape[2:])

            x = torch.cat([
                x,
                self.skip(center_crop(skip, skip.shape[2:])),
            ],
                          dim=1)

        out_fmap = self.decode(x)
        return out_fmap, self.to_rgb(out_fmap) + rgb


class UNetBone(nn.Module):
    """
    Configurable UNet model.

    Note: Not all input sizes are valid. Make sure that the model can decode an
    image of the same size first.

    Args:
        arch (list): an architecture specification made of:
            - an int, for an kernel with specified output_channels
            - 'U' for upsampling+conv
            - 'D' for downsampling (maxpooling)
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """
    @tu.experimental
    def __init__(self,
                 arch,
                 in_ch=3,
                 out_ch=1,
                 *,
                 skip=True,
                 up_mode='bilinear'):
        super(UNetBone, self).__init__()
        self.in_conv = nn.Sequential(
            tu.kaiming(nn.Conv2d(in_ch, arch[0], 5, padding=2)),
            nn.LeakyReLU(LEAK, True)
            )
        self.conv = nn.Sequential(
            tch.nn.ResBlock(arch[-1], arch[-1], norm=None),
            #tch.nn.SelfAttention2d(arch[-1]),
            tch.nn.ResBlock(arch[-1], arch[-1], norm=None),
        )
        for x1, x2 in zip(reversed(arch[:-1]), reversed(arch[1:])):
            self.conv = UBlock(x1, x2, self.conv, skip=skip, up_mode=up_mode)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (tensor): input tensor, batch of images
        """
        out = self.conv(self.in_conv(x))
        return torch.sigmoid(out[1])


@tu.experimental
def make_net(in_sz=256, n_down=4, max_ch=128, base_ch=8):
    layers = []
    for _ in range(n_down):
        layers.append(min(base_ch, max_ch))
        assert in_sz // 2 * 2 == in_sz
        in_sz = in_sz // 2
        base_ch *= 2

    model = UNetBone(layers, 3, 3, skip=False)
    return model


class PerceptualDiscriminator(nn.Module):
    @tu.experimental
    def __init__(self, n_downscales=4):
        super().__init__()


@tu.experimental
def cutblur(base, patcher):
    a = min(base.shape[2], base.shape[3]) - 1
    s = random.randrange(int(a/5), int(a/3))

    y = random.randrange(0, base.shape[2] - 1 - s)
    x = random.randrange(0, base.shape[3] - 1 - s)

    if random.uniform(0, 1) < 0.5:
        base[:, :, y:y + s, x:x + s] = patcher[:, :, y:y + s, x:x + s]
    else:
        clone = patcher.clone()
        clone[:, :, y:y + s, x:x + s] = base[:, :, y:y + s, x:x + s]
        base = clone

    return base

class RandomZoomNoLoss:
    @tu.experimental
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        a = min(img.size)
        if self.size > a:
            print('Warning, RandomZoomNoLoss will zoom with loss because '
                    'target size ' + str(self.size) + ' > img size ' +
                    str(img.size))
            return img
        a = random.randrange(self.size, a)
        return TF.functional.resize(img, a, interpolation=Image.BICUBIC)

class PairedFrames:
    @tu.experimental
    def __init__(self, root1, root2, transform=None):
        self.ds1 = tch.datasets.FastImageFolder(root1)
        self.ds2 = tch.datasets.FastImageFolder(root2)
        assert len(self.ds1) == len(self.ds2)
        self.transform = transform

    def __getitem__(self, i):
        x1, _ = self.ds1[i]
        x2, _ = self.ds2[i]
        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        return x1, x2

    def __len__(self):
        return len(self.ds1)

@tu.experimental
def train(rank, world_size):
    from torchvision.datasets import ImageFolder
    import torchelie.loss.gan.standard as gan_loss
    from torchelie.loss.gan.penalty import zero_gp
    scale = 1
    G = make_net(base_ch=64, max_ch=512, n_down=int(6))

    for name, m in G.named_modules():

        if isinstance(m, tch.nn.PreactResBlock):
            m.relu = nn.LeakyReLU(0.2, True)
            m.branch.relu = nn.LeakyReLU(0.2, True)
            m.branch.relu2 = nn.LeakyReLU(0.2, True)

            if 'decode' in name and False:
                layers = [f for f in m.branch.named_children()]
                c = m.branch.conv1.weight.shape[0]
                layers[3] = ('noise1', tch.nn.Noise(c, inplace=True, bias=False))
                c = m.branch.conv2.weight.shape[0]
                #layers[7] = ('noise2', tch.nn.Noise(c, inplace=True, bias=False))
                m.branch = nn.Sequential(OrderedDict(layers))

    for name, m in G.named_modules():
        if isinstance(m, nn.Conv2d) and not 'rgb' in name:
            m.padding_mode = PAD_MODE
            nn.init.normal_(m.weight)

            def l2_norm_w(w):
                # kaiming scaling
                w = w * math.sqrt(2 / w.shape[1])
                # l2 norm from StyleGAN2
                return w / (w.view(w.shape[0], -1).norm(dim=1,
                    keepdim=True).view(-1, 1, 1, 1) + 1e-6)

            tch.nn.utils.weight_lambda(m, 'l2', l2_norm_w)

        if isinstance(m, nn.Conv2d) and 'rgb' in name:
            m.padding_mode = PAD_MODE
            tu.kaiming(m, a=LEAK, dynamic=True)

    for name, m in G.named_modules():
        if isinstance(m, tch.nn.PreactResBlock):
            pass#m.branch.conv2.weight_g.data.zero_()

    D = tch.models.StyleGAN2Discriminator(128,
                                          input_ch=6,
                                          ch_mul=1 / 4,
                                          with_minibatch_std=False).bone
    D.add_module(
        'out',
        tu.kaiming(tch.nn.Conv3x3(D.conv4x4.branch[-1].weight.shape[0], 1),
                   dynamic=True))

    if rank == 0:
        print(G)
        print(D)

    if True:
        ckpt = torch.load('model/ckpt_3000.pth', map_location='cpu')
        print(G.load_state_dict(ckpt['G'], strict=False))
        print(D.load_state_dict(ckpt['D'], strict=False))
        del ckpt
    G = nn.parallel.DistributedDataParallel(G.to(rank),
                                            device_ids=[rank],
                                            output_device=rank)
    D = nn.parallel.DistributedDataParallel(D.to(rank),
                                            device_ids=[rank],
                                            output_device=rank)

    ada = True
    diffTF = ADATF(-2 if not ada else -0.6, 0.005)

    def gblur(x, k):
        if k == 3:
            w = [[
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]
            ]]
        else:
            w = [[
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ]]
        w = torch.tensor(w, device=x.device) / (16 if k == 3 else 256)
        w = torch.stack([w for _ in range(x.shape[1])], dim=0)
        p = k//2
        x = F.pad(x, (p, p, p, p), mode='reflect')
        x = F.conv2d(x, w, groups=x.shape[1])
        return x

    def downgrade(x):
        x_down = x.clone()

        if random.choice([True, False, False, False]):
            k = random.choice([3, 5])
            x_down = gblur(x_down, k)

        down_mode = random.choice(['bicubic'])
        factor = random.choice([1, 1.5, 2, 3])
        x_down = F.interpolate(
            x_down,
            scale_factor=1 / factor,
            mode=down_mode,
            align_corners=False)

        #x_down = gblur(x_down, 3)

        x_down = F.interpolate(x_down,
                               size=x.shape[2:],
                               mode='bicubic',
                               align_corners=False)

        return x_down

    ii = 00
    g_norm = 0

    def D_fun(batch):
        nonlocal ii
        nonlocal g_norm
        ii += 1
        (x, x_down) = batch[0]
        if random.uniform(0, 1) < 0.9:
            pass#x_down = downgrade(x_down)
        real_x = torch.cat(diffTF(x_down, x), dim=1) * 2 - 1

        with torch.no_grad():
            with G.no_sync():
                out = G(x_down * 2 - 1)
        out.requires_grad_(True)
        out.retain_grad()
        fake_x = torch.cat(diffTF(x_down, out), dim=1) * 2 - 1

        real_out = D(real_x)
        pos_ratio = real_out.gt(0).float().mean().cpu().item()
        print('pos_ratio', pos_ratio, 'p', diffTF.p)
        #diffTF.log_loss(-pos_ratio)

        real_loss = gan_loss.real(real_out)
        with D.no_sync():
            real_loss.backward()

        if ii % 16 == 0:
            with D.no_sync():

                class Scaler:
                    def is_enabled(self):
                        return True

                    def scale(self, x):
                        return x / 1000

                    def get_scale(self):
                        return 1 / 1000

                gp, g_norm = zero_gp(D, real_x, fake_x.detach(), Scaler())
                (0.1 * gp).backward()

        fake_pred = D(fake_x)
        fake_loss = gan_loss.fake(fake_pred)
        fake_loss.backward()

        fake_grad = out.grad.detach().norm(dim=1, keepdim=True)
        fake_grad /= fake_grad.max()
        return {
            'loss': (real_loss + fake_loss).item(),
            'g_norm': g_norm,
            'grad_norm': g_norm,
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'down': x_down[:2],
            'imgs': out[:2].detach().cpu(),
        }

    ploss = tch.loss.PerceptualLoss(['conv2_2', 'conv3_4'],
                                    rescale=True, remove_unused_layers=True,
                                    use_avg_pool=True).to(rank)

    def G_fun(batch):
        (x, x_down) = batch[0]
        if random.uniform(0, 1) < 0.9:
            pass#x_down = downgrade(x_down)
        #x_down = cutblur(x_down, x)
        out = G(x_down * 2 - 1)
        out.requires_grad_(True)
        out.retain_grad()

        out_g = out.detach().clone()
        out_g.requires_grad_(True)
        out_g.retain_grad()

        loss = 0.03 * ploss(out_g, x)
        loss.backward()

        d_in = torch.cat(diffTF(x_down, out_g), dim=1) * 2 - 1
        d_out = D(d_in)
        loss = gan_loss.generated(d_out)
        loss.backward()

        out.backward(out_g.grad)

        out_grad = out.grad.detach().norm(dim=1, keepdim=True)
        out_grad /= out_grad.max()
        return {
            'G_loss': loss.item(),
            'up': out[:2].detach(),
            'down': x_down[:2].detach(),
            'truth': x[:2].detach(),
            'grad': out_grad[:2],
        }

    tfm = TF.Compose([
        tch.transforms.ResizeNoCrop(int(scale * 854), mode=Image.BICUBIC),
        TF.CenterCrop((int(scale * 480), int(scale * 854))),
        TF.ToTensor()
    ])
    dst = tch.datasets.NoexceptDataset(
        PairedFrames(
        '/hdd/data/resources/SR/angela/140p/',
        '/hdd/data/resources/SR/angela/lol/',
                                     transform=tfm))
    import itertools
    dlt = iter(itertools.cycle(torch.utils.data.DataLoader(dst,
                                     num_workers=4,
                                     shuffle=True,
                                     pin_memory=True,
                                     drop_last=True,
                                     batch_size=1)))
    def test_fun(batch):
        _, x = next(dlt)
        x = x.to(rank)
        with torch.no_grad():
            out = G(x * 2 - 1)
        return {'HQ_test': out[:2].detach(), 'LQ_test': x[:2].detach()}


    tfm = TF.Compose([
        RandomZoomNoLoss(int(scale*854)),
        TF.RandomCrop((int(scale * 480), int(scale * 854)), pad_if_needed=True, padding_mode='reflect'),
        TF.RandomHorizontalFlip(),
        tch.transforms.MultiBranch([
            TF.ToTensor(),
            TF.Compose([
                TF.RandomChoice([
                    TF.Resize(240, Image.BICUBIC),
                    TF.Resize(180, Image.BICUBIC),
                    TF.Resize(120, Image.BICUBIC),
                ]),
                jpeg_compress,
                TF.Resize((480, 854), Image.BICUBIC),
                TF.ToTensor(),
            ])
        ])
    ])
    ds = tch.datasets.NoexceptDataset(
            tch.datasets.FastImageFolder('/hdd/data/resources/pos//',
                                     transform=tfm))
    # BYPASS
    if False:
        tfm = TF.Compose([
            tch.transforms.ResizeNoCrop(854, mode=Image.BICUBIC),
            TF.CenterCrop((int(480), int(854))),
            TF.ToTensor()
        ])
        ds = tch.datasets.NoexceptDataset(
            PairedFrames(
            '/hdd/data/resources/SR/480p/',
            '/hdd/data/resources/SR/240p/',
                                         transform=tfm))
    dl = torch.utils.data.DataLoader(ds,
                                     num_workers=4,
                                     shuffle=True,
                                     pin_memory=True,
                                     drop_last=True,
                                     batch_size=1)
    recipe = GANRecipe(G,
                       D,
                       G_fun,
                       D_fun,
                       test_fun,
                       dl,
                       visdom_env='main' if rank == 0 else None,
                       checkpoint='model' if rank == 0 else None,
                       test_every=100,
                       log_every=10)
    recipe.G_loop.callbacks.add_callbacks([
        tch.callbacks.Log('up', 'up'),
        tch.callbacks.Log('down', 'down'),
        tch.callbacks.Log('truth', 'truth'),
        tch.callbacks.Log('grad', 'grad'),
        tch.callbacks.Optimizer(
            tch.optim.Lookahead(
            tch.optim.RAdamW(G.parameters(),
                             lr=1e-3,
                             betas=(0., 0.99),
                             weight_decay=0)
            , k=10)
        ),
    ])
    recipe.callbacks.add_callbacks([
        tch.callbacks.WindowedMetricAvg('real_loss'),
        tch.callbacks.WindowedMetricAvg('fake_loss'),
        tch.callbacks.WindowedMetricAvg('grad_norm'),
        tch.callbacks.Optimizer(
            tch.optim.Lookahead(
            tch.optim.RAdamW(D.parameters(),
                             lr=1e-3,
                             betas=(0., 0.99),
                             weight_decay=0)
            , k=10)
        ),
    ])
    recipe.test_loop.callbacks.add_callbacks([
        tch.callbacks.Log('HQ_test', 'HQ_test'),
        tch.callbacks.Log('LQ_test', 'LQ_test'),
    ])
    recipe.to(rank)
    recipe.run(50)


if __name__ == '__main__':
    tch.utils.parallel_run(train)
