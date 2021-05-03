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


@tu.experimental
def gradient_penalty_M(model, fake, ins, real, objective_norm: float):
    fake = fake.detach()
    ins = ins.detach()

    t = torch.rand(fake.shape[0], 1, 1, 1, device=fake.device)
    fake = t * fake + (1 - t) * real

    fake.requires_grad_(True)
    ins.requires_grad_(True)

    out = F.softmax(model(fake, ins)['matches'], dim=1).sum()

    g = torch.autograd.grad(outputs=out,
                            inputs=fake,
                            create_graph=True,
                            only_inputs=True)[0]

    g_norm = g.pow(2).sum(dim=(1, 2, 3)).add_(1e-8).sqrt()
    return (g_norm - objective_norm).pow(2).mean(), g_norm.mean().item()


class Matcher(nn.Module):
    @tu.experimental
    def __init__(self):
        super().__init__()
        self.net = res_discr_3l()
        self.net.classifier = nn.Sequential()
        self.net.to_equal_lr()
        self.proj_size = 256
        self.proj_A = tu.kaiming(tnn.Conv1x1(128, self.proj_size), dynamic=True)
        self.proj_B = tu.kaiming(tnn.Conv1x1(128, self.proj_size), dynamic=True)

    def forward(self, fake, ins):
        # fake = F.interpolate(fake, scale_factor=0.5, mode='bilinear')
        # ins = F.interpolate(ins, scale_factor=0.5, mode='bilinear')
        f1 = self.proj_A(self.net(fake))
        f2 = self.proj_B(self.net(ins))
        n, c, h, w = f1.shape
        out = torch.bmm(
            f1.permute(2, 3, 0, 1).reshape(-1, n, c),
            f2.permute(2, 3, 0, 1).reshape(-1, n, c).permute(0, 2, 1)
        )
        out = out.view(h, w, n, n).permute(2, 3, 0, 1)

        N = len(out)
        labels = torch.arange(N, device=out.device)
        labels = labels.view(N, 1, 1).expand(N, h, w)
        return {'matches': out,
                'loss': F.cross_entropy(out, labels),
                'labels': labels}


def Crop(x):
    return x.crop((20, 0, 220, 220))


@tu.experimental
def celeba(male, tfm=None):
    from torchvision.datasets import CelebA
    celeba = CelebA('~/.torch/celeba', download=True, target_type=[])
    male_idx = celeba.attr_names.index('Male')
    files = [
        f'~/.torch/celeba/celeba/img_align_celeba/{celeba.filename[i]}'
        for i in range(len(celeba))
        if celeba.attr[i, male_idx] == (1 if male else 0)
    ]
    return tch.datasets.pix2pix.ImagesPaths(files, tfm)


@tu.experimental
def train(rank, world_size):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data-A', required=True)
    parser.add_argument('--data-B', required=True)
    parser.add_argument('--r0-D', default=0.0001, type=float)
    parser.add_argument('--r0-M', default=0.0001, type=float)
    parser.add_argument('--consistency', default=0.01, type=float)
    parser.add_argument('--from-ckpt')
    opts = parser.parse_args()

    G = pix2pix_256()
    D = res_discr_4l()
    D.add_minibatch_stddev()
    D.to_equal_lr()
    M = Matcher()

    for m in G.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # tu.kaiming(m, a=0.2, dynamic=True)
            weight_norm_and_equal_lr(m, 0.2)
    G.remove_batchnorm()

    if rank == 0:
        print(G)
        print(D)
        print(M)

    G = torch.nn.parallel.DistributedDataParallel(G.to(rank), [rank], rank)
    D = torch.nn.parallel.DistributedDataParallel(D.to(rank), [rank], rank)
    M = torch.nn.parallel.DistributedDataParallel(M.to(rank), [rank], rank)

    SIZE = 128
    if False:
        ds_A = UnlabeledImages(
            opts.data_A,
            TF.Compose([
                TF.Resize(SIZE),
                TF.CenterCrop(SIZE),
                TF.RandomHorizontalFlip(),
                TF.ToTensor(),
            ]))
        ds_B = UnlabeledImages(
            opts.data_B,
            TF.Compose([
                Crop,
                TF.Resize(SIZE),
                TF.CenterCrop(SIZE),
                TF.RandomHorizontalFlip(),
                TF.ToTensor(),
            ]))
    else:
        ds_A = celeba(
            True,
            TF.Compose([
                TF.Resize(SIZE),
                TF.CenterCrop(SIZE),
                TF.RandomHorizontalFlip(),
                TF.ToTensor(),
            ]))
        ds_B = celeba(
            False,
            TF.Compose([
                TF.Resize(SIZE),
                TF.CenterCrop(SIZE),
                TF.RandomHorizontalFlip(),
                TF.ToTensor(),
            ]))
    print('ds', len(ds_A), len(ds_B))
    ds = tch.datasets.RandomPairsDataset(ds_A, ds_B)

    ds = torch.utils.data.DataLoader(ds,
                                     8,
                                     num_workers=4,
                                     drop_last=True,
                                     shuffle=True,
                                     pin_memory=True)

    def G_fun(batch) -> dict:
        x, y = batch
        D.eval()
        out = G(x * 2 - 1)
        with D.no_sync():
            loss = gan_loss.generated(D(out * 2 - 1))
        loss.backward(retain_graph=True)

        with M.no_sync():
            clf_loss = opts.consistency * M(out * 2 - 1, x * 2 - 1)['loss']

        # labels = torch.arange(len(matches), device=matches.device)
        # clf_loss = opts.consistency * F.cross_entropy( matches, torch.arange(len(matches), device=matches.device))
        clf_loss.backward()
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

    def D_fun(batch) -> dict:
        G.eval()
        x, y = batch
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
            gp, match_g_norm = gradient_penalty_M(M, fake, x * 2 - 1, real, 0)
            (opts.r0_M * gp).backward()

        M_out = M(fake, x * 2 - 1)
        matches = M_out['matches']
        labels = M_out['labels']
        match_correct = matches.argmax(1).eq(labels).float().mean()

        loss = M_out['loss']
        loss.backward()

        return {
            'out': out,
            'fake_loss': fake_loss.item(),
            'prob_fake': torch.sigmoid(prob_fake).mean().item(),
            'prob_real': torch.sigmoid(prob_real).mean().item(),
            'real_loss': real_loss.item(),
            'g_norm': g_norm,
            'D-correct':
            (fake_correct + real_correct) / (2 * prob_fake.numel()),
            'match_correct': match_correct,
            'match_g_norm': match_g_norm,
        }

    def test_fun(_):
        return {}

    recipe = GANRecipe(G,
                       D,
                       G_fun,
                       D_fun,
                       test_fun,
                       ds,
                       checkpoint='face_inpaint' if rank == 0 else None,
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
                             betas=(0., 0.99),
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
    recipe.to(rank)
    if opts.from_ckpt is not None:
        recipe.load_state_dict(torch.load(opts.from_ckpt, map_location='cpu'))
    recipe.run(200)


if __name__ == '__main__':
    tu.parallel_run(train)
