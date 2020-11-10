import random
import math
import torch
import torch.nn as nn
import torchelie as tch
import torchelie.utils as tu
import torchelie.callbacks as tcb
import torchelie.nn as tnn
from torchelie.recipes.gan import GANRecipe
from collections import OrderedDict
from typing import Optional

class ADATF:
    def __init__(self, target_loss: float):
        self.p = 0
        self.target_loss = target_loss

    def __call__(self, x):
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

    def log_loss(self, l):
        if l > self.target_loss:
            self.p -= 0.01
        else:
            self.p += 0.01
        self.p = max(0, min(self.p, 1))


class PPL:
    def __init__(self, model, noise_size, batch_size, device, every=4):
        self.model = model
        self.batch_size = batch_size
        self.every = every
        self.noise_size = noise_size
        self.device = device

    def on_batch_start(self, state):
        if state['iters'] % self.every != 0:
            return
        with self.model.no_sync():
            tu.unfreeze(self.model)
            with torch.enable_grad():
                ppl = self.model.module.ppl(torch.randn(self.batch_size,
                                               self.noise_size,
                                               device=self.device))
                (self.every * ppl).backward()
                state['ppl'] = ppl.item()


def train(rank, world_size):
    from torchelie.models import StyleGAN2Generator, StyleGAN2Discriminator
    import torchelie.loss.gan.standard as gan_loss
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as TF
    from torchelie.optim import RAdamW, Lookahead
    from torchelie.loss.gan.penalty import zero_gp
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--noise-size', type=int, default=128)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--ch-mul', type=float, default=1.)
    parser.add_argument('--max-ch', type=int, default=512)
    parser.add_argument('--no-ada', action='store_true')
    parser.add_argument('--from-ckpt')
    opts = parser.parse_args()

    tag = opts.img_dir.split('/')[-1] or opts.img_dir.split('/')[-2]

    G = StyleGAN2Generator(opts.noise_size,
                           img_size=opts.img_size,
                           ch_mul=opts.ch_mul,
                           max_ch=opts.max_ch, equal_lr=True)
    G_polyak = StyleGAN2Generator(opts.noise_size,
                                  img_size=opts.img_size,
                                  ch_mul=opts.ch_mul,
                                  max_ch=opts.max_ch, equal_lr=True)
    D = StyleGAN2Discriminator(input_sz=opts.img_size,
                               ch_mul=opts.ch_mul,
                               max_ch=opts.max_ch, equal_lr=True)

    with torch.no_grad():
        for gp, g in zip(G_polyak.parameters(), G.parameters()):
            gp.copy_(g)

    G = nn.parallel.DistributedDataParallel(G.to(rank), [rank], rank)
    D = nn.parallel.DistributedDataParallel(D.to(rank), [rank], rank)
    print(G)
    print(D)

    optG = Lookahead(
            RAdamW(G.parameters(), 2e-3, betas=(0., 0.99), weight_decay=0),
            k=10)
    optD = Lookahead(
            RAdamW(D.parameters(), 4e-3, betas=(0., 0.99), weight_decay=0),
            k=10)

    tfm = TF.Compose([
        tch.transforms.ResizeNoCrop(int(opts.img_size * 1.1)),
        tch.transforms.AdaptPad(
            (int(1.1 * opts.img_size), int(1.1 * opts.img_size)),
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
    diffTF = ADATF(-2 if opts.no_ada else -0.6)

    gam = 0.1


    def G_train(batch):
        ##############
        ### G pass ###
        ##############
        imgs = G(torch.randn(opts.batch_size, opts.noise_size,
                             device=rank))
        pred = D(diffTF(imgs) * 2 - 1)
        score = gan_loss.generated(pred)
        score.backward()

        return {'G_loss': score.item()}

    ii = 0

    g_norm = 0
    def D_train(batch):
        nonlocal ii
        nonlocal gam
        nonlocal g_norm

        ###################
        #### Fake pass ####
        ###################
        with D.no_sync():
            # Sync the gradient on the last backward
            noise = torch.randn(opts.batch_size,
                                opts.noise_size,
                                device=rank)
            with torch.no_grad():
                fake = G(noise)
            fake.requires_grad_(True)
            fake.retain_grad()
            fake_tf = diffTF(fake) * 2 - 1
            fakeness = D(fake_tf).squeeze(1)
            fake_loss = gan_loss.fake(fakeness)
            fakeness = fakeness.argsort(0)
            fake_loss.backward()

            correct = (fakeness < 0).int().eq(1).float().sum()
        fake_grad = fake.grad.detach().norm(dim=1, keepdim=True)
        fake_grad /= fake_grad.max()

        tfmed = diffTF(batch[0]) * 2 - 1

        ##############
        #### 0-GP ####
        ##############
        if ii % 16 == 0:
            with D.no_sync():
                gp, g_norm = zero_gp(D, tfmed.detach_(), fake_tf.detach_())
                # Sync the gradient on the next backward
                (16 * gam * gp).backward()
                gam = max(1e-6, gam / 1.1) if g_norm < 0.3 else gam * 1.1
        ii += 1

        ###################
        #### Real pass ####
        ###################
        real_out = D(tfmed)
        correct += (real_out > 0).detach().int().eq(1).float().sum()
        real_loss = gan_loss.real(real_out)
        real_loss.backward()
        pos_ratio = real_out.gt(0).float().mean().cpu()
        diffTF.log_loss(-pos_ratio)
        return {
            'imgs': fake.detach()[fakeness],
            'i_grad': fake_grad[fakeness],
            'loss': real_loss.item() + fake_loss.item(),
            'fake_loss': fake_loss.item(),
            'real_loss': real_loss.item(),
            'grad_norm': g_norm,
            'ADA-p': diffTF.p,
            'gamma': gam,
            'D-correct': correct / (2 * opts.batch_size),
        }

    tu.freeze(G_polyak)

    def test(batch):
        G_polyak.eval()
        def sample(N, n_iter, alpha=0.01, show_every=10):
            noise = torch.randn(N,
                                opts.noise_size,
                                device=rank,
                                requires_grad=True)
            opt = torch.optim.Adam([noise], lr=alpha)
            fakes = []
            for i in range(n_iter):
                noise += torch.randn_like(noise) / 10
                fake_batch = []
                opt.zero_grad()
                for j in range(0, N, opts.batch_size):
                    with torch.enable_grad():
                        n_batch = noise[j:j + opts.batch_size]
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

        noise1 = torch.randn(
            opts.batch_size * 2 // 8, 1, opts.noise_size, device=rank)
        noise2 = torch.randn(
            opts.batch_size * 2 // 8, 1, opts.noise_size, device=rank)
        t = torch.linspace(0, 1, 8, device=noise1.device).view(8, 1)
        noise = noise1 * t + noise2 * (1 - t)
        noise = noise.view(-1, opts.noise_size)
        interp = torch.cat([
            G_polyak(n, mixing=False)
            for n in torch.split(noise, opts.batch_size)
        ],
                           dim=0)
        return {
            'polyak_imgs': fake,
            'polyak_interp': interp,
        }

    recipe = GANRecipe(G,
                       D,
                       G_train,
                       D_train,
                       test,
                       dl,
                       visdom_env='gan_' + tag if rank == 0 else None,
                       log_every=10,
                       test_every=1000,
                       checkpoint='gan_' + tag if rank == 0 else None,
                       g_every=1)
    recipe.callbacks.add_callbacks([
        tcb.Log('batch.0', 'x'),
        tcb.WindowedMetricAvg('fake_loss'),
        tcb.WindowedMetricAvg('real_loss'),
        tcb.WindowedMetricAvg('grad_norm'),
        tcb.WindowedMetricAvg('ADA-p'),
        tcb.WindowedMetricAvg('gamma'),
        tcb.WindowedMetricAvg('D-correct'),
        tcb.Log('i_grad', 'img_grad'),
        tch.callbacks.Optimizer(optD),
    ])
    recipe.G_loop.callbacks.add_callbacks([
        tch.callbacks.Optimizer(optG),
        PPL(G, opts.noise_size, opts.batch_size // 2, rank, every=4),
        tcb.Polyak(G, G_polyak, 0.5**((opts.batch_size*world_size)/20000)),
        tcb.WindowedMetricAvg('ppl'),
    ])
    recipe.test_loop.callbacks.add_callbacks([
        tcb.Log('polyak_imgs', 'polyak'),
        tcb.Log('polyak_interp', 'interp'),
    ])
    recipe.register('G_polyak', G_polyak)
    if opts.from_ckpt is not None:
        ckpt = torch.load(opts.from_ckpt, map_location='cpu')
        print('G_polyak:', G_polyak.load_state_dict(ckpt['G_polyak'], strict=False))
        print('G:', G.module.load_state_dict(ckpt['G'], strict=False))
        print('D:', D.module.load_state_dict(ckpt['D'], strict=False))
    recipe.to(rank)
    recipe.run(5000)


if __name__ == '__main__':
    tu.parallel_run(train)
