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
"""
MtF:
torchelie.recipes.unpaired
        --r0-D 0.01
        --r0-M 0.1
        --consistency 1

"""


@tu.experimental
class GradientPenaltyM:

    def __init__(self, gamma):
        self.gamma = gamma
        self.iters = 0
        self.last_norm = float('nan')

    def do(self, model, fake_dst, src, real_dst, objective_norm: float):
        fake_dst = fake_dst.detach()
        src = src.detach()

        t = torch.rand(fake_dst.shape[0], 1, 1, 1, device=fake_dst.device)
        fake_dst = t * fake_dst + (1 - t) * real_dst

        fake_dst.requires_grad_(True)
        src.requires_grad_(True)

        out = model(fake_dst, src)['matches'].sum()

        g = torch.autograd.grad(outputs=out,
                                inputs=fake_dst,
                                create_graph=True,
                                only_inputs=True)[0]

        g_norm = g.pow(2).sum(dim=(1, 2, 3)).add_(1e-8).sqrt()
        return (g_norm - objective_norm).pow(2).mean(), g_norm.mean().item()

    def __call__(self, model, fake_dst, src, real_dst):
        if self.iters < 100 or self.iters % 4 == 0:
            real_dst = real_dst.detach()
            fake_dst = fake_dst.detach()
            gp, g_norm = self.do(model, fake_dst, src, real_dst, 0)
            # Sync the gradient on the next backward
            if torch.any(torch.isnan(gp)):
                gp.detach_()
            else:
                (4 * self.gamma * gp).backward()
            self.last_norm = g_norm
        self.iters += 1
        return self.last_norm


class Matcher(nn.Module):

    @tu.experimental
    def __init__(self, n_scales=2):
        super().__init__()
        proj_size = 256

        self.n_scales = n_scales
        self.nets = nn.ModuleDict()
        self.proj_As = nn.ModuleDict()
        self.proj_Bs = nn.ModuleDict()
        for i in range(n_scales):
            net = residual_patch70()
            net.classifier = nn.Sequential()
            net.to_equal_lr()
            proj_A = nn.Sequential(
                nn.LeakyReLU(0.2, False),
                tu.kaiming(tnn.Conv1x1(256, proj_size), dynamic=True))
            proj_B = nn.Identity()

            self.nets[str(i)] = net
            self.proj_As[str(i)] = proj_A
            self.proj_Bs[str(i)] = proj_B

    def cross_entropy(self, f1, f2):
        n, c, h, w = f1.shape
        out = torch.bmm(
            f1.permute(2, 3, 0, 1).reshape(-1, n, c),
            f2.permute(2, 3, 0, 1).reshape(-1, n, c).permute(0, 2, 1))
        out = out.view(h, w, n, n).permute(2, 3, 0, 1)

        N = len(out)
        labels = torch.arange(N, device=out.device)
        labels = labels.view(N, 1, 1).expand(N, h, w)
        return F.cross_entropy(out, labels)

    def barlow(self, f1, f2):
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        n, c, h, w = f1.shape
        out = torch.bmm(
            f1.permute(2, 3, 0, 1).reshape(-1, n, c),
            f2.permute(2, 3, 0, 1).reshape(-1, n, c).permute(0, 2, 1))
        out = out.view(h, w, n, n).permute(2, 3, 0, 1)

        labels = torch.eye(n, device=out.device)
        labels = labels.view(n, n, 1, 1).expand(n, n, h, w)
        return out, F.smooth_l1_loss(out, labels, beta=0.1, reduction='sum')

    def bce(self, f1, f2):
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        n, c, h, w = f1.shape
        out = torch.bmm(
            f1.permute(2, 3, 0, 1).reshape(-1, n, c),
            f2.permute(2, 3, 0, 1).reshape(-1, n, c).permute(0, 2, 1))
        out = out.view(h, w, n, n).permute(2, 3, 0, 1)

        labels = torch.eye(n, device=out.device)
        labels = labels.view(n, n, 1, 1).expand(n, n, h, w)
        return out, F.binary_cross_entropy_with_logits(out, labels)

    def forward(self, fake, ins):
        total_loss = 0
        outs = []
        all_labels = []
        for scale_order in range(self.n_scales):
            scale = 2**scale_order
            fake_scale = F.interpolate(fake,
                                       scale_factor=1 / scale,
                                       mode='bilinear')
            ins_scale = F.interpolate(ins,
                                      scale_factor=1 / scale,
                                      mode='bilinear')

            f1 = self.proj_As[str(scale_order)](
                self.nets[str(scale_order)](fake_scale))
            f2 = self.proj_Bs[str(scale_order)](
                self.nets[str(scale_order)](ins_scale))
            N, c, h, w = f1.shape
            labels = torch.arange(N, device=f1.device)
            labels = labels.view(N, 1, 1).expand(N, h, w)

            out, loss = self.barlow(f1, f2)
            total_loss += loss
            outs.append(out.reshape(out.shape[0], out.shape[1], -1))
            all_labels.append(labels.reshape(labels.shape[0], -1))
        outs = torch.cat(outs, dim=2)
        return {
            'matches': outs,
            'loss': total_loss / outs.numel(),
            'labels': torch.cat(all_labels, dim=1)
        }


def Crop(x):
    return x.crop((20, 0, 220, 220))


def get_dataset(typ: str, path: str, train: bool, size: int):
    if typ == 'images':
        return UnlabeledImages(
            path,
            TF.Compose(([Crop] if 'trainA' in path else []) + [
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
    G = pix2pix_256()
    G.to_instance_norm()
    tnn.utils.net_to_equal_lr(G, leak=0.2)

    D = residual_patch70()
    tnn.utils.net_to_equal_lr(D, leak=0.2)
    D = MultiScaleDiscriminator(D)

    M = Matcher()

    if rank == 0:
        print(G)
        print(D)
        print(M)

    G = torch.nn.parallel.DistributedDataParallel(G.to(rank), [rank], rank)
    D = torch.nn.parallel.DistributedDataParallel(D.to(rank), [rank], rank)
    M = torch.nn.parallel.DistributedDataParallel(M.to(rank), [rank], rank)

    SIZE = 128
    ds_A = get_dataset(opts.data_A[0], opts.data_A[1], True, SIZE)
    ds_B = get_dataset(opts.data_B[0], opts.data_B[1], True, SIZE)

    ds_test = get_dataset(opts.data_test[0], opts.data_test[1], False, SIZE)

    print('ds', len(ds_A), len(ds_B))
    ds = tch.datasets.RandomPairsDataset(ds_A, ds_B)

    ds = torch.utils.data.DataLoader(ds,
                                     8,
                                     num_workers=4,
                                     drop_last=True,
                                     shuffle=True,
                                     pin_memory=True)

    ds_test = torch.utils.data.DataLoader(ds_test,
                                          16,
                                          num_workers=4,
                                          drop_last=True,
                                          shuffle=True,
                                          pin_memory=True)

    fake_out = [None]

    def G_fun(batch) -> dict:
        x, y = batch
        D.train()
        M.eval()
        out = G(x * 2 - 1)
        out_d = out.detach()
        out_d.requires_grad_()
        with D.no_sync():
            loss = gan_loss.generated(D(out_d * 2 - 1))
        loss.backward()

        with M.no_sync():
            clf_loss = opts.consistency * M(out_d * 2 - 1, x * 2 - 1)['loss']

        clf_loss.backward()
        out.backward(out_d.grad)
        fake_out[0] = out.detach()
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

    gradient_penalty = GradientPenalty(opts.r0_D)
    gradient_penalty_M = GradientPenaltyM(opts.r0_M)

    def D_fun(batch) -> dict:
        G.train()
        x, y = batch
        M.train()
        with G.no_sync():
            with torch.no_grad():
                out = G(x * 2 - 1)
        fake = out * 2 - 1
        real = y * 2 - 1
        with D.no_sync():
            prob_fake = D(fake)
            fake_correct = prob_fake.detach().lt(0).int().eq(1).sum()
            fake_loss = gan_loss.fake(prob_fake)
            fake_loss.backward()

        with D.no_sync():
            g_norm = gradient_penalty(D, real, fake)

        prob_real = D(real)
        real_correct = prob_real.detach().gt(0).int().eq(1).sum()
        real_loss = gan_loss.real(prob_real)
        real_loss.backward()

        with M.no_sync():
            match_g_norm = gradient_penalty_M(M, fake, x * 2 - 1, real)

        M_out = M(fake, x * 2 - 1)
        matches = M_out['matches']
        labels = M_out['labels']
        match_correct = matches.argmax(1).eq(labels).float().mean()

        loss = M_out['loss']
        loss.backward()

        return {
            'out':
                out,
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
            'match_correct':
                match_correct,
            'match_g_norm':
                match_g_norm,
        }

    def test_fun(x):
        G.train()
        with torch.no_grad():
            return {'out': G(x * 2 - 1).detach()}

    recipe = GANRecipe(G,
                       D,
                       G_fun,
                       D_fun,
                       test_fun,
                       ds,
                       test_loader=ds_test,
                       checkpoint='main' if rank == 0 else None,
                       visdom_env='main' if rank == 0 else None)
    recipe.register('M', M)

    recipe.callbacks.add_callbacks([
        tch.callbacks.Optimizer(
            tch.optim.RAdamW(D.parameters(),
                             lr=2e-3,
                             betas=(0., 0.99),
                             weight_decay=0)),
        tch.callbacks.Optimizer(
            tch.optim.RAdamW(M.parameters(),
                             lr=2e-3,
                             betas=(0.9, 0.99),
                             weight_decay=0)),
        tch.callbacks.Log('out', 'out'),
        tch.callbacks.Log('batch.0', 'x'),
        tch.callbacks.Log('batch.1', 'y'),
        # tch.callbacks.Log('batch.0.1', 'y'),
        tch.callbacks.WindowedMetricAvg('fake_loss', 'fake_loss'),
        tch.callbacks.WindowedMetricAvg('real_loss', 'real_loss'),
        tch.callbacks.WindowedMetricAvg('prob_fake', 'prob_fake'),
        tch.callbacks.WindowedMetricAvg('prob_real', 'prob_real'),
        tch.callbacks.WindowedMetricAvg('D-correct', 'D-correct'),
        tch.callbacks.WindowedMetricAvg('match_correct', 'match_correct'),
        tch.callbacks.Log('g_norm', 'g_norm'),
        tch.callbacks.Log('match_g_norm', 'match_g_norm'),
    ])
    recipe.G_loop.callbacks.add_callbacks([
        tch.callbacks.Optimizer(
            tch.optim.RAdamW(G.parameters(),
                             lr=2e-3,
                             betas=(0., 0.99),
                             weight_decay=0)),
    ])
    recipe.test_loop.callbacks.add_callbacks([
        tch.callbacks.Log('out', 'test_out'),
        tch.callbacks.Log('batch', 'test_x'),
    ])
    recipe.to(rank)
    if opts.from_ckpt is not None:
        recipe.load_state_dict(torch.load(opts.from_ckpt, map_location='cpu'))
    recipe.run(200)


def run(opts):
    G = pix2pix_256()
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
    TF.functional.to_pil_image(G(img)[0]).save(opts.dst)


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
    train_parser.set_defaults(func=lambda opts: tu.parallel_run(train, opts))

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--from-ckpt', required=True)
    run_parser.add_argument('--src', required=True)
    run_parser.add_argument('--dst', required=True)
    run_parser.set_defaults(func=run)

    opts = parser.parse_args()
    opts.func(opts)
