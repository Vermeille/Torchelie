import torch
import torchelie as tch
import torchelie.utils as tu
from torchelie.recipes.gan import GANRecipe
import torchvision.transforms as TF
import torchelie.loss.gan.standard as gan_loss
from torchelie.loss.gan.penalty import zero_gp
from torchelie.datasets.pix2pix import UnlabeledImages
from torchelie.models import *
import torch.nn as nn


class Matcher(nn.Module):

    @tu.experimental
    def __init__(self, n_scales=3):
        super().__init__()
        proj_size = 256

        self.n_scales = n_scales
        self.nets = nn.ModuleDict()
        self.projs = nn.ModuleDict()
        for i in range(n_scales):
            net = patch34().remove_batchnorm()
            net.classifier = nn.Sequential()
            net.to_equal_lr()
            proj = nn.Sequential(
                nn.LeakyReLU(0.2, False),
                tu.kaiming(tnn.Conv1x1(proj_size, proj_size), dynamic=True))

            self.nets[str(i)] = net
            self.projs[str(i)] = proj

    def barlow(self, src, proj):
        src = F.normalize(src, dim=1)
        proj = F.normalize(proj, dim=1)
        n, c, h, w = src.shape
        out = torch.bmm(
            src.permute(2, 3, 0, 1).reshape(-1, n, c),
            proj.permute(2, 3, 0, 1).reshape(-1, n, c).permute(0, 2, 1))
        out = out.view(h, w, n, n).permute(2, 3, 0, 1)

        labels = torch.eye(n, device=out.device)
        labels = labels.view(n, n, 1, 1).expand(n, n, h, w)
        return {
            'cosine': out,
            'loss': F.smooth_l1_loss(out, labels, beta=0.1),
            'src_feats': src,
            'proj_feats': proj
        }

    def forward(self, src, dst):
        outs = []
        for scale_order in range(self.n_scales):
            scale = 2**scale_order
            src_scale = F.interpolate(src,
                                      scale_factor=1 / scale,
                                      mode='bilinear')
            dst_scale = F.interpolate(dst,
                                      scale_factor=1 / scale,
                                      mode='bilinear')

            src_feats = self.nets[str(scale_order)](src_scale)
            proj_feats = self.projs[str(scale_order)](
                self.nets[str(scale_order)](dst_scale))
            N, c, h, w = src_feats.shape
            labels = torch.arange(N, device=src.device)
            labels = labels.view(N, 1, 1).expand(N, h, w)

            outs.append(self.barlow(src_feats, proj_feats))
            outs[-1]['labels'] = labels

        total_loss = sum(out['loss'] for out in outs)
        matches = torch.cat([
            out['cosine'].view(out['cosine'].shape[0], out['cosine'].shape[1],
                               -1) for out in outs
        ],
                            dim=2)
        all_labels = torch.cat(
            [out['labels'].view(out['labels'].shape[0], -1) for out in outs],
            dim=1)
        return {
            'matches': matches,
            'loss': total_loss,
            'labels': all_labels,
            'src_feats': [out['src_feats'] for out in outs],
            'proj_feats': [out['proj_feats'] for out in outs],
        }


def get_dataset(typ: str, path: str, train: bool, size: int):
    if typ == 'images':
        return UnlabeledImages(
            path,
            TF.Compose([
                TF.Resize(size),
                TF.CenterCrop(size),
                TF.RandomHorizontalFlip(),
                TF.ToTensor(),
            ]))
    if typ == 'celeba':
        return celeba(
            path, train,
            TF.Compose([
                TF.Resize(size),
                TF.CenterCrop(size),
                TF.RandomHorizontalFlip(),
                TF.ToTensor(),
            ]))


@tu.experimental
def celeba(path, train: bool, tfm=None):
    from torchvision.datasets import CelebA
    positive = True
    if path[:4] == 'not-':
        positive = False
        path = path[4:]
    celeba = CelebA('~/.torch/celeba',
                    download=True,
                    target_type=[],
                    split='train' if train else 'test')
    male_idx = celeba.attr_names.index(path)
    files = [
        f'~/.torch/celeba/celeba/img_align_celeba/{celeba.filename[i]}'
        for i in range(len(celeba))
        if celeba.attr[i, male_idx] == (1 if positive else 0)
    ]
    return tch.datasets.pix2pix.ImagesPaths(files, tfm)


@tu.experimental
def train(rank, world_size, opts):

    def make_G():
        G = pix2pix_128()
        G.to_instance_norm()

        def to_adain(m):
            if isinstance(m, nn.InstanceNorm2d):
                #return tnn.AdaIN2d(m.num_features, 256)
                return tnn.FiLM2d(m.num_features, 256)
            return m

        tnn.edit_model(G, to_adain)
        tnn.utils.net_to_equal_lr(G, leak=0.2)
        return G

    Gx = make_G()
    Gy = make_G()

    def make_D():
        D = patch34()
        D.set_input_specs(3 + 3)
        D.remove_batchnorm()
        tnn.utils.net_to_equal_lr(D, leak=0.2)
        #D.features[0].conv.weight_g.data.normal_(0, 0.02)
        D = MultiScaleDiscriminator(D)
        return D

    Dx = make_D()
    Dy = make_D()

    if rank == 0:
        print(Gx)
        print(Dx)

    Gx = torch.nn.parallel.DistributedDataParallel(Gx.to(rank), [rank], rank)
    Gy = torch.nn.parallel.DistributedDataParallel(Gy.to(rank), [rank], rank)
    Dx = torch.nn.parallel.DistributedDataParallel(Dx.to(rank), [rank], rank)
    Dy = torch.nn.parallel.DistributedDataParallel(Dy.to(rank), [rank], rank)

    SIZE = 128
    ds_A = get_dataset(opts.data_A[0], opts.data_A[1], True, SIZE)
    ds_B = get_dataset(opts.data_B[0], opts.data_B[1], True, SIZE)

    ds = tch.datasets.RandomPairsDataset(ds_A, ds_B)

    ds_test_A = get_dataset(opts.data_test[0], opts.data_test[1], False, SIZE)
    ds_test_B = get_dataset(opts.data_B[0], opts.data_B[1], False, SIZE)
    ds_test = tch.datasets.RandomPairsDataset(ds_test_A, ds_test_B)

    if rank == 0:
        print(ds)
        print(ds_test)

    ds = torch.utils.data.DataLoader(ds,
                                     8,
                                     num_workers=4,
                                     drop_last=True,
                                     shuffle=True,
                                     pin_memory=True)

    ds_test = torch.utils.data.DataLoader(ds_test,
                                          128,
                                          num_workers=4,
                                          drop_last=True,
                                          shuffle=True,
                                          pin_memory=True)

    def dpo(val, p=0):
        if torch.rand(1).item() < p:
            return torch.zeros_like(val)
        else:
            return val

    def G_fun(batch) -> dict:
        x, y = batch
        out = Gy(x * 2 - 1, torch.randn(x.shape[0], 256, device=x.device))
        with Dy.no_sync():
            loss = gan_loss.generated(
                Dy(torch.cat([out * 2 - 1, x * 2 - 1], dim=1)))
        with Dx.no_sync():
            pass
            #loss += gan_loss.fake(Dx(torch.cat([x * 2 - 1, out * 2 - 1], dim=1)))
        loss.backward()

        out = Gx(y * 2 - 1, torch.randn(x.shape[0], 256, device=x.device))
        with Dx.no_sync():
            loss = gan_loss.generated(
                Dx(torch.cat([out * 2 - 1, y * 2 - 1], dim=1)))
        with Dy.no_sync():
            pass
            #loss += gan_loss.fake(Dy(torch.cat([y * 2 - 1, out * 2 - 1], dim=1)))
        loss.backward()

        return {'G_loss': loss.item()}

    class GradientPenalty:

        def __init__(self, gamma):
            self.gamma = gamma
            self.iters = 0
            self.last_norm = float('nan')

        def __call__(self, model, real, fake):
            if self.iters < 100 or self.iters % 4 == 0:
                real = real.detach()
                fake = fake.detach()
                gp, g_norm = zero_gp(model, real, fake)
                # Sync the gradient on the next backward
                if torch.any(torch.isnan(gp)):
                    gp.detach_()
                else:
                    (4 * self.gamma * gp).backward()
                self.last_norm = g_norm
            self.iters += 1
            return self.last_norm

    gradient_penalty_x = GradientPenalty(opts.r0_D)
    gradient_penalty_y = GradientPenalty(opts.r0_D)

    def D_fun(batch) -> dict:
        x, y = batch
        x = x * 2 - 1
        y = y * 2 - 1

        with torch.no_grad():
            with Gy.no_sync():
                out = Gy(x, torch.randn(x.shape[0], 256, device=x.device))
            y_ = out * 2 - 1
            with Gx.no_sync():
                out = Gx(y, torch.randn(x.shape[0], 256, device=x.device))
            x_ = out * 2 - 1

        neg = [x_, y]

        with Dx.no_sync():
            prob_fake = Dx(torch.cat(neg, dim=1))
            fake_correct = prob_fake.detach().lt(0).int().eq(1).sum()
            fake_loss = gan_loss.fake(prob_fake)
            fake_loss.backward()

        with Dx.no_sync():
            g_norm = gradient_penalty_x(Dx, torch.cat([x, dpo(y_)], dim=1),
                                        torch.cat(neg, dim=1))
        prob_real = Dx(torch.cat([x, dpo(y_)], dim=1))
        real_correct = prob_real.detach().gt(0).int().eq(1).sum()
        real_loss = gan_loss.real(prob_real)
        real_loss.backward()

        neg = [y_, x]

        with Dy.no_sync():
            prob_fake = Dy(torch.cat(neg, dim=1))
            fake_correct = prob_fake.detach().lt(0).int().eq(1).sum()
            fake_loss = gan_loss.fake(prob_fake)
            fake_loss.backward()

        with Dy.no_sync():
            g_norm = gradient_penalty_y(Dy, torch.cat([y, dpo(x_)], dim=1),
                                        torch.cat(neg, dim=1))

        prob_real = Dy(torch.cat([y, dpo(x_)], dim=1))
        real_correct = prob_real.detach().gt(0).int().eq(1).sum()
        real_loss = gan_loss.real(prob_real)
        real_loss.backward()

        return {
            'out':
                torch.cat([x_, y_], dim=0),
            'fake_loss':
                fake_loss.item(),
            'prob_fake':
                torch.sigmoid(prob_fake).mean().item(),
            'prob_real':
                torch.sigmoid(prob_real).mean().item(),
            'real_loss':
                real_loss.item(),
            'g_norm':
                g_norm,
            'D-correct':
                (fake_correct + real_correct) / (2 * prob_fake.numel()),
        }

    def test_fun(batch):
        x, y = batch
        with Gy.no_sync():
            out_y = torch.cat([
                Gy(
                    xx * 2 - 1,
                    tch.distributions.sample_truncated_normal(
                        xx.shape[0], 256).to(xx.device))
                for xx in torch.split(x, 32)
            ],
                              dim=0)
        with Gx.no_sync():
            out_x = torch.cat([
                Gx(
                    yy * 2 - 1,
                    tch.distributions.sample_truncated_normal(
                        yy.shape[0], 256).to(yy.device))
                for yy in torch.split(y, 32)
            ],
                              dim=0)
            return {'out': torch.cat([out_x, out_y])}

    recipe = GANRecipe(nn.ModuleList([Gx, Gy]),
                       nn.ModuleList([Dx, Dy]),
                       G_fun,
                       D_fun,
                       test_fun,
                       ds,
                       test_loader=ds_test,
                       test_every=5000,
                       log_every=100,
                       checkpoint='main_adain' if rank == 0 else None,
                       visdom_env='main_adain' if rank == 0 else None)

    recipe.callbacks.add_callbacks([
        tch.callbacks.Optimizer(
            tch.optim.Lookahead(
                tch.optim.RAdamW(Dx.parameters(),
                                 lr=2e-3,
                                 betas=(0., 0.99),
                                 weight_decay=0))),
        tch.callbacks.Optimizer(
            tch.optim.Lookahead(
                tch.optim.RAdamW(Dy.parameters(),
                                 lr=2e-3,
                                 betas=(0., 0.99),
                                 weight_decay=0))),
        tch.callbacks.Log('out', 'out'),
        tch.callbacks.Log('batch.0', 'x'),
        tch.callbacks.Log('batch.1', 'y'),
        #tch.callbacks.Log('batch.1', 'y'),
        # tch.callbacks.Log('batch.0.1', 'y'),
        #tch.callbacks.WindowedMetricAvg('fake_loss', 'fake_loss'),
        #tch.callbacks.WindowedMetricAvg('real_loss', 'real_loss'),
        #tch.callbacks.WindowedMetricAvg('prob_fake', 'prob_fake'),
        #tch.callbacks.WindowedMetricAvg('prob_real', 'prob_real'),
        tch.callbacks.WindowedMetricAvg('D-correct', 'D-correct'),
        tch.callbacks.Log('g_norm', 'g_norm'),
    ])
    recipe.G_loop.callbacks.add_callbacks([
        tch.callbacks.Optimizer(
            tch.optim.Lookahead(
                tch.optim.RAdamW(Gx.parameters(),
                                 lr=2e-3,
                                 betas=(0., 0.99),
                                 weight_decay=0))),
        tch.callbacks.Optimizer(
            tch.optim.Lookahead(
                tch.optim.RAdamW(Gy.parameters(),
                                 lr=2e-3,
                                 betas=(0., 0.99),
                                 weight_decay=0))),
    ])
    recipe.test_loop.callbacks.add_callbacks([
        tch.callbacks.GANMetrics('batch.1', 'out', device=rank),
        tch.callbacks.Log('kid', 'kid'),
        tch.callbacks.Log('fid', 'fid'),
        tch.callbacks.Log('precision', 'precision'),
        tch.callbacks.Log('recall', 'recall'),
        tch.callbacks.Log('out', 'test_out'),
        tch.callbacks.Log('batch.0', 'test_x'),
    ])
    recipe.to(rank)
    if opts.from_ckpt is not None:
        recipe.load_state_dict(torch.load(opts.from_ckpt, map_location='cpu'))
    recipe.run(200)


def run(opts):
    G = pix2pix_128()
    G.to_instance_norm()
    tnn.utils.net_to_equal_lr(G, leak=0.2)
    G.load_state_dict(torch.load(opts.from_ckpt, map_location='cpu')['G'])

    import torchvision.transforms as TF
    from PIL import Image
    tfm = TF.Compose([
        TF.Resize(128),
        TF.CenterCrop(128),
        TF.ToTensor(),
        TF.Normalize([0.5] * 3, [0.5] * 3),
    ])
    img = tfm(Image.open(opts.src).convert('RGB'))
    img = torch.stack([img, img], dim=0)
    TF.functional.to_pil_image(G(img, torch.randn(2, 256))[0]).save(opts.dst)


def para_run(opts):
    return tu.parallel_run(train, opts=opts)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--data-A',
                              required=True,
                              type=lambda x: x.split(':'))
    train_parser.add_argument('--data-B',
                              required=True,
                              type=lambda x: x.split(':'))
    train_parser.add_argument('--data-test',
                              required=True,
                              type=lambda x: x.split(':'))
    train_parser.add_argument('--r0-D', default=0.0001, type=float)
    train_parser.add_argument('--r0-M', default=0.0001, type=float)
    train_parser.add_argument('--consistency', default=0.01, type=float)
    train_parser.add_argument('--from-ckpt')

    train_parser.set_defaults(func=para_run)

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--from-ckpt', required=True)
    run_parser.add_argument('--src', required=True)
    run_parser.add_argument('--dst', required=True)
    run_parser.set_defaults(func=run)

    opts = parser.parse_args()
    opts.func(opts)
