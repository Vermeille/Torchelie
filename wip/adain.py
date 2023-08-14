import torch
import torch.nn as nn
import torch.nn.functional as F
import torchelie.nn as tnn
import torchelie.utils as tu
import torch.utils.checkpoint as cp
#from pytorch_memlab import profile
import collections
from typing import Optional, List, Union, Tuple


def std_s(x, dim: Union[int, Tuple[int, ...]]) -> torch.Tensor:
    return torch.sqrt(x.var(dim, keepdim=True) + 1e-8)


class AdaIN(nn.Module):
    z_std: nn.Module
    z_m: nn.Module

    @tu.experimental
    def __init__(self, ch: int) -> None:
        super(AdaIN, self).__init__()
        self.z_std = nn.Linear(ch * 2, ch)
        tu.xavier(self.z_std)
        self.z_std.weight.data.zero_()
        self.z_m = nn.Linear(ch * 2, ch)
        tu.xavier(self.z_m)
        self.z_m.weight.data.zero_()

    def condition(self, y: torch.Tensor) -> None:
        self.ystd = std_s(y, (2, 3))
        self.ym = y.mean((2, 3), keepdim=True)
        f = torch.cat([self.ystd, self.ym], dim=1)[:, :, 0, 0]
        self.zs = self.z_std(f)[:, :, None, None]
        self.zm = self.z_m(f)[:, :, None, None]

    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if y is not None:
            self.condition(y)
        del y

        x = x - x.mean((2, 3), keepdim=True)
        x = x.div(std_s(x, (2, 3)))
        N, C, H, W = x.shape
        x.add_(
            torch.randn(N, 1, H, W,
                        device=x.device).mul(self.zs).add_(self.zm))
        return x.mul_(self.ystd).add_(self.ym)


@tu.experimental
def make_nets():
    from torchelie.models import PaddedPerceptualNet
    m = PaddedPerceptualNet([])
    m.model = m.model[:30]
    print(m.model)
    m.set_keep_layers(names=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])

    dconv = []
    for nm, l in m.model.named_children():
        if 'pad' in nm:
            pass  # ignore
        elif 'conv' in nm:
            dconv.append((
                nm,
                #tu.kaiming(
                nn.Conv2d(l.out_channels, l.in_channels, l.kernel_size)))
            #)
            dconv.append((nm.replace('conv', 'pad'), nn.ReflectionPad2d(1)))
        elif 'pool' in nm:
            dconv.append((nm, nn.UpsamplingNearest2d(scale_factor=2)))
        elif 'relu' in nm:
            dconv.append((nm, nn.ReLU(True)))
    dconv = list(reversed(dconv))

    if 'relu' in dconv[0][0]:
        dconv = dconv[1:]

    d = nn.Sequential(collections.OrderedDict(dconv))
    from torchelie.data_learning import CorrelateColors
    #d.add_module('color', CorrelateColors())
    d.add_module('sigmoid', nn.Sigmoid())
    print(d)
    return m, d


import random
from PIL import Image
import torchvision as tv


class Dataset:
    @tu.experimental
    def __init__(self, path, l=2, ctfm=None, stfm=None):
        ds = tv.datasets.ImageFolder(path)
        iidx = ds.class_to_idx['content']
        self.images = [s[0] for s in ds.samples if s[1] == iidx]
        sidx = ds.class_to_idx['style']
        self.styles = [s[0] for s in ds.samples if s[1] == sidx]
        self.styles = self.images
        ds = tv.datasets.ImageFolder('/hdd/data/mscoco')
        self.images += [s[0] for s in ds.samples]
        self.ctfm = ctfm
        self.stfm = stfm
        self.l = l

    def __len__(self):
        return self.l

    def __getitem__(self, i):
        content = Image.open(random.choice(self.images)).convert('RGB')
        sidx = random.randrange(0, len(self.styles))
        style = Image.open(self.styles[sidx]).convert('RGB')

        if self.ctfm is not None:
            content = self.ctfm(content)

        if self.stfm is not None:
            style = self.stfm(style)

        return content, style, sidx


class Rotate90:
    def __call__(self, x):
        angle = random.choice([0, 90, 180, 270])
        return tv.transforms.functional.rotate(x, angle)


class Stylizer(nn.Module):
    @tu.experimental
    def __init__(self):
        super(Stylizer, self).__init__()
        enc, dec = make_nets()
        tu.freeze(enc)

        self.enc = enc
        self.dec = dec
        self.adain = AdaIN(512)
        self.norm = tnn.ImageNetInputNorm()

    def condition(self, s):
        es = self.enc(self.norm(s), detach=True)[0]
        self.adain.condition(es)

    def stylize(self, c, s=None, ratio=0.9):
        if s is not None:
            self.condition(s)

        ec = self.enc(self.norm(c), detach=True)[0]
        t = self.adain(ec)

        ratio_w = ratio.view(-1, 1, 1, 1)
        rep = ratio_w * t + (1 - ratio_w) * ec
        del ec
        return self.dec(rep)

    def forward(self, c, s, ratio=0.9):
        with torch.no_grad():
            es, s_acts = self.enc(self.norm(s), detach=True)
            self.adain.condition(es)
            del es

            ec = self.enc(self.norm(c), detach=True)[0]

        ratio[:] = 0.5
        t = self.adain(ec)
        ratio_w = ratio.view(-1, 1, 1, 1)
        rep = t  #ratio_w * t + (1 - ratio_w) * ec

        img = self.dec(rep)

        reenc = self.enc(self.norm(img), detach=False)[1]

        losses = {}
        content_loss = F.l1_loss(reenc['relu4_1'], t, reduction='none').mean(
            (1, 2, 3))

        style_loss = 0
        for l in s_acts.keys():
            style_loss = style_loss + F.mse_loss(s_acts[l].mean((2, 3)),
                                                 reenc[l].mean((2, 3)),
                                                 reduction='none').mean(1)

            style_loss = style_loss + F.mse_loss(std_s(s_acts[l], (2, 3)),
                                                 std_s(reenc[l], (2, 3)),
                                                 reduction='none').mean(1)
        #return img, torch.mean((1 - ratio) * content_loss + ratio * style_loss)
        W = 0.7
        losses['style_loss'] = torch.mean(W * style_loss).item()
        losses['content_loss'] = torch.mean((1 - W) * content_loss).item()
        losses['loss'] = torch.mean(W * style_loss + (1 - W) * content_loss)
        return img, losses


@tu.experimental
def train(rank, world_size):
    net = Stylizer()
    net = net.to(rank)
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[rank],
                                              find_unused_parameters=True)
    net.load_state_dict(torch.load('model/ckpt_80000.pth',
                                   map_location='cuda:' + str(rank))['model'],
                        strict=False)

    def forward_train(batch):
        c, s, sidx = batch
        ratio = torch.rand(c.shape[0]).to(c.device)
        img, losses = net(c, s, ratio)
        losses['loss'].backward()
        losses['out'] = img
        return losses

    def forward_test(batch):
        c, s, sidx = batch

        n_steps = 8
        n_samples = c.shape[0]
        c = torch.repeat_interleave(c, n_steps, dim=0)
        s = s.repeat_interleave(n_steps, dim=0)
        r = torch.empty(n_samples, n_steps)
        r[:] = torch.linspace(0, 1, n_steps)
        r = r.to(c.device)
        r = r.view(-1)
        out = net.module.stylize(c, s, ratio=r)
        return {'out': out}

    import random

    small = tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(256, scale=(0.01, 1)),
        Rotate90(),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomVerticalFlip(),
        tv.transforms.ColorJitter(0.6, 0.6, 0.2, 0.1),
        tv.transforms.ToTensor()
    ])
    big = tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(512, scale=(0.01, 1)),
        Rotate90(),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomVerticalFlip(),
        tv.transforms.ColorJitter(0.5, 0.5, 0.2, 0.1),
        tv.transforms.ToTensor()
    ])
    big_clean = tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(512, scale=(0.01, 1)),
        tv.transforms.ToTensor()
    ])
    small_clean = tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(512, scale=(0.01, 1)),
        tv.transforms.ToTensor()
    ])
    small_clean_fixed = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(256),
        tv.transforms.ToTensor()
    ])
    import torchelie as tch

    ds = torch.utils.data.DataLoader(
        Dataset(sys.argv[1], l=10005, ctfm=big, stfm=small),
        batch_size=4,
        drop_last=True,
        #pin_memory=True,
        num_workers=6,
        shuffle=True)
    dst = torch.utils.data.DataLoader(
        Dataset(sys.argv[1], l=5, ctfm=small_clean, stfm=small_clean_fixed),
        batch_size=3,
        #pin_memory=True,
        drop_last=True,
        num_workers=2)

    recipe = TrainAndTest(net,
                          forward_train,
                          forward_test,
                          ds,
                          dst,
                          log_every=50,
                          visdom_env='style_{}'.format(rank),
                          checkpoint='model' if rank == 0 else None,
                          test_every=500)
    import torchelie.callbacks as tcb
    params = [(x.numel(), nm) for nm, x in net.named_parameters()]
    params.sort(key=lambda x: -x[0])

    def sz(n):
        if n >= 1024 * 1024 * 1024:
            return str(n //
                       (1024 * 1024 * 1024)) + 'G ' + sz(n %
                                                         (1024 * 1024 * 1024))
        if n >= 1024 * 1024:
            return str(n // (1024 * 1024)) + 'M ' + sz(n % (1024 * 1024))
        if n >= 1024:
            return str(n // (1024)) + 'k ' + sz(n % (1024))
        return str(n)

    print([(nm, sz(p * 4)) for p, nm in params[:10]])
    LR = 1e-4
    opt = tch.optim.RAdamW(net.parameters(), lr=LR, weight_decay=0)
    #opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    recipe.register('opt', opt)
    recipe.callbacks.add_callbacks([
        tcb.WindowedMetricAvg('loss'),
        tcb.WindowedMetricAvg('style_loss'),
        tcb.WindowedMetricAvg('content_loss'),
        tcb.Log('batch.1', 'style'),
        tcb.Log('batch.0', 'content'),
        tcb.Log('out', 'out'),
        tcb.Optimizer(opt, clip_grad_norm=10, log_lr=True),
        tcb.LRSched(torch.optim.lr_scheduler.ReduceLROnPlateau(opt))
    ])
    recipe.test_loop.callbacks.add_callbacks([
        tcb.Log('out', 'out'),
        tcb.Log('batch.1', 'style'),
        tcb.Log('batch.0', 'content'),
    ])
    recipe.to(rank)
    #torch.autograd.set_detect_anomaly(True)
    recipe.run(1000000)




if __name__ == '__main__':
    tu.parallel_run(train)
