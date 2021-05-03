import random
import copy
import math
import torch
import torch.nn as nn
import torchelie as tch
import torchelie.utils as tu
import torchelie.callbacks as tcb
import torchelie.loss.gan.standard as gan_loss
from torchelie.optim import RAdamW
from torchelie.loss.gan.penalty import zero_gp
from torchelie.recipes.gan import GANRecipe
from torchelie.callbacks.avg import ExponentialAvg


@tu.experimental
class ADATF:
    p: float

    def __init__(self, target_loss: float, growth: float = 0.01):
        self.p = 0
        self.target_loss = target_loss
        self.growth = growth
        self.loss = ExponentialAvg(0.95)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0:
            return x
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

        geom = tch.transforms.differentiable.AllAtOnceGeometric(x.shape[0])
        geom.translate(0.25, 0.25, p)
        geom.rotate(180, p)
        geom.scale(0.5, 0.5, p)
        geom.flip_x(0.5, p)
        geom.flip_y(0.5, p)
        x = geom.apply(x)
        return x

    def log_loss(self, loss: float) -> None:
        self.loss.log(loss)
        if self.loss.get() > self.target_loss:
            self.p -= self.growth
        else:
            self.p += self.growth
        self.p = max(0, min(self.p, 0.9))


class PPL:
    @tu.experimental
    def __init__(self, every=4):
        self.every = every
        self.ppl_goal = 0.
        self.iters = 0

    def compute_ppl(self, model, z):
        gen, w = model(z, get_w=True, mixing=False)
        B, C, H, W = gen.shape
        noise = torch.randn_like(gen) / math.sqrt(H * W)
        JwTy = torch.autograd.grad(outputs=torch.sum(gen * noise),
                                   inputs=w,
                                   create_graph=True,
                                   only_inputs=True)[0]
        JwTy_norm = JwTy.pow(2).sum(1).mul(
            1 / (2 * len(model.module.render))).sqrt()

        E_JwTy_norm = JwTy_norm.detach().mean().item()
        if self.ppl_goal is None:
            self.ppl_goal = E_JwTy_norm
        self.ppl_goal = 0.99 * self.ppl_goal + 0.01 * E_JwTy_norm

        return (JwTy_norm - self.ppl_goal).pow(2).mean() * 2

    def __call__(self, model, z):
        ppl = self.ppl_goal
        if self.iters % self.every == 0:
            ppl = self.compute_ppl(model, z)
            (self.every * ppl).backward()

        self.iters += 1
        return self.ppl_goal


class GradientPenalty:
    @tu.experimental
    def __init__(self, gamma):
        self.gamma = gamma
        self.iters = 0
        self.last_norm = float('nan')

    def __call__(self, model, real, fake):
        if self.iters % 4 == 3:
            real = real.detach()
            fake = fake.detach()
            gp, g_norm = zero_gp(model, real, fake)
            # Sync the gradient on the next backward
            if not torch.any(torch.isnan(gp)):
                (4 * self.gamma * gp).backward()
            self.last_norm = g_norm
        self.iters += 1
        return self.last_norm


@tu.experimental
def StyleGAN2Recipe(G: nn.Module,
                    D: nn.Module,
                    dataloader,
                    noise_size: int,
                    gpu_id: int,
                    total_num_gpus: int,
                    *,
                    G_lr: float = 2e-3,
                    D_lr: float = 2e-3,
                    tag: str = 'model',
                    ada: bool = True):
    """
    StyleGAN2 Recipe distributed with DistributedDataParallel

    Args:
        G (nn.Module): a Generator.
        D (nn.Module): a Discriminator.
        dataloader: a dataloader conforming to torchvision's API.
        noise_size (int): the size of the input noise vector.
        gpu_id (int): the GPU index on which to run.
        total_num_gpus (int): how many GPUs are they
        G_lr (float): RAdamW lr for G
        D_lr (float): RAdamW lr for D
        tag (str): tag for Visdom and checkpoints
        ada (bool): whether to enable Adaptive Data Augmentation

    Returns:
        recipe, G EMA model
    """
    G_polyak = copy.copy(G)

    G = nn.parallel.DistributedDataParallel(G.to(gpu_id), [gpu_id], gpu_id)
    D = nn.parallel.DistributedDataParallel(D.to(gpu_id), [gpu_id], gpu_id)
    print(G)
    print(D)

    optG: torch.optim.Optimizer = RAdamW(G.parameters(),
                                         G_lr,
                                         betas=(0., 0.99),
                                         weight_decay=0)
    optD: torch.optim.Optimizer = RAdamW(D.parameters(),
                                         D_lr,
                                         betas=(0., 0.99),
                                         weight_decay=0)

    batch_size = len(next(iter(dataloader))[0])
    diffTF = ADATF(-2 if not ada else -0.9,
                   50000 / (batch_size * total_num_gpus))

    ppl = PPL(4)

    def G_train(batch):
        with G.no_sync():
            pl = ppl(G, torch.randn(batch_size, noise_size, device=gpu_id))
        ##############
        #   G pass   #
        ##############
        imgs = G(torch.randn(batch_size, noise_size, device=gpu_id))
        pred = D(diffTF(imgs) * 2 - 1)
        score = gan_loss.generated(pred)
        score.backward()

        return {'G_loss': score.item(), 'ppl': pl}

    gradient_penalty = GradientPenalty(0.1)

    def D_train(batch):
        ###################
        #    Fake pass    #
        ###################
        with D.no_sync():
            # Sync the gradient on the last backward
            noise = torch.randn(batch_size, noise_size, device=gpu_id)
            with torch.no_grad():
                fake = G(noise)
            fake.requires_grad_(True)
            fake.retain_grad()
            fake_tf = diffTF(fake) * 2 - 1
            fakeness = D(fake_tf).squeeze(1)
            fake_loss = gan_loss.fake(fakeness)
            fake_loss.backward()

            correct = (fakeness < 0).int().eq(1).float().sum()
        fake_grad = fake.grad.detach().norm(dim=1, keepdim=True)
        fake_grad /= fake_grad.max()

        tfmed = diffTF(batch[0]) * 2 - 1

        with D.no_sync():
            grad_norm = gradient_penalty(D, batch[0] * 2 - 1,
                                         fake.detach() * 2 - 1)

        ###################
        #    Real pass    #
        ###################
        real_out = D(tfmed)
        correct += (real_out > 0).detach().int().eq(1).float().sum()
        real_loss = gan_loss.real(real_out)
        real_loss.backward()
        pos_ratio = real_out.gt(0).float().mean().cpu().item()
        diffTF.log_loss(-pos_ratio)
        return {
            'imgs': fake.detach(),
            'i_grad': fake_grad,
            'loss': real_loss.item() + fake_loss.item(),
            'fake_loss': fake_loss.item(),
            'real_loss': real_loss.item(),
            'ADA-p': diffTF.p,
            'D-correct': correct / (2 * real_out.numel()),
            'grad_norm': grad_norm
        }

    tu.freeze(G_polyak)

    def test(batch):
        G_polyak.eval()

        def sample(N, n_iter, alpha=0.01, show_every=10):
            noise = torch.randn(N,
                                noise_size,
                                device=gpu_id,
                                requires_grad=True)
            opt = torch.optim.Adam([noise], lr=alpha)
            fakes = []
            for i in range(n_iter):
                noise += torch.randn_like(noise) / 10
                fake_batch = []
                opt.zero_grad()
                for j in range(0, N, batch_size):
                    with torch.enable_grad():
                        n_batch = noise[j:j + batch_size]
                        fake = G_polyak(n_batch, mixing=False)
                        fake_batch.append(fake)
                        log_prob = n_batch[:, 32:].pow(2).mul_(-0.5)
                        fakeness = -D(fake * 2 - 1).sum() - log_prob.sum()
                        fakeness.backward()
                opt.step()
                fake_batch = torch.cat(fake_batch, dim=0)

                if i % show_every == 0:
                    fakes.append(fake_batch.cpu().detach().clone())

            fakes.append(fake_batch.cpu().detach().clone())

            return torch.cat(fakes, dim=0)

        fake = sample(8, 50, alpha=0.001, show_every=10)

        noise1 = torch.randn(batch_size * 2 // 8, 1, noise_size, device=gpu_id)
        noise2 = torch.randn(batch_size * 2 // 8, 1, noise_size, device=gpu_id)
        t = torch.linspace(0, 1, 8, device=noise1.device).view(8, 1)
        noise = noise1 * t + noise2 * (1 - t)
        noise = noise.view(-1, noise_size)
        interp = torch.cat([
            G_polyak(n, mixing=False) for n in torch.split(noise, batch_size)
        ], dim=0)
        return {
            'polyak_imgs': fake,
            'polyak_interp': interp,
        }

    recipe = GANRecipe(G,
                       D,
                       G_train,
                       D_train,
                       test,
                       dataloader,
                       visdom_env=tag if gpu_id == 0 else None,
                       log_every=10,
                       test_every=1000,
                       checkpoint=tag if gpu_id == 0 else None,
                       g_every=1)
    recipe.callbacks.add_callbacks([
        tcb.Log('batch.0', 'x'),
        tcb.WindowedMetricAvg('fake_loss'),
        tcb.WindowedMetricAvg('real_loss'),
        tcb.WindowedMetricAvg('grad_norm'),
        tcb.WindowedMetricAvg('ADA-p'),
        tcb.WindowedMetricAvg('D-correct'),
        tcb.Log('i_grad', 'img_grad'),
        tch.callbacks.Optimizer(optD),
    ])
    recipe.G_loop.callbacks.add_callbacks([
        tch.callbacks.Optimizer(optG),
        tcb.Polyak(G.module, G_polyak,
                   0.5**((batch_size * total_num_gpus) / 20000)),
        tcb.WindowedMetricAvg('ppl'),
    ])
    recipe.test_loop.callbacks.add_callbacks([
        tcb.Log('polyak_imgs', 'polyak'),
        tcb.Log('polyak_interp', 'interp'),
    ])
    recipe.register('G_polyak', G_polyak)
    recipe.to(gpu_id)
    return recipe, G_polyak


@tu.experimental
def train(rank, world_size):
    from torchelie.models import StyleGAN2Generator, StyleGAN2Discriminator
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as TF
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--noise-size', type=int, default=128)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--ch-mul', type=float, default=1.)
    parser.add_argument('--max-ch', type=int, default=512)
    parser.add_argument('--no-ada', action='store_true')
    parser.add_argument('--gp-thresh', type=float, default=0.3)
    parser.add_argument('--from-ckpt')
    opts = parser.parse_args()

    tag = opts.img_dir.split('/')[-1] or opts.img_dir.split('/')[-2]
    tag = 'gan_' + tag

    G = StyleGAN2Generator(opts.noise_size,
                           img_size=opts.img_size,
                           ch_mul=opts.ch_mul,
                           max_ch=opts.max_ch)
    D = StyleGAN2Discriminator(input_sz=opts.img_size,
                               ch_mul=opts.ch_mul,
                               max_ch=opts.max_ch)

    tfm = TF.Compose([
        tch.transforms.ResizeNoCrop(opts.img_size),
        tch.transforms.AdaptPad((opts.img_size, opts.img_size),
                                padding_mode='edge'),
        TF.Resize(opts.img_size),
        TF.RandomHorizontalFlip(),
        TF.ToTensor()
    ])
    ds = tch.datasets.NoexceptDataset(ImageFolder(opts.img_dir, transform=tfm))
    dl = torch.utils.data.DataLoader(ds,
                                     num_workers=4,
                                     shuffle=True,
                                     pin_memory=True,
                                     drop_last=True,
                                     batch_size=opts.batch_size)
    recipe, _ = StyleGAN2Recipe(G,
                                D,
                                dl,
                                opts.noise_size,
                                rank,
                                world_size,
                                tag=tag,
                                ada=not opts.no_ada)
    if opts.from_ckpt is not None:
        ckpt = torch.load(opts.from_ckpt, map_location='cpu')
        recipe.load_state_dict(ckpt)

    recipe.run(5000)


if __name__ == '__main__':
    tu.parallel_run(train)
