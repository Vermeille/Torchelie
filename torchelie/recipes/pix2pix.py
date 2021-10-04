from typing import Tuple
import copy
import os

import torch
import torch.nn as nn
import torchvision.transforms as TF

import torchelie as tch
import torchelie.utils as tu
from torchelie.recipes.gan import GANRecipe
from torchelie.transforms import MultiBranch
from torchelie.loss.gan.penalty import zero_gp
from torchelie.datasets import UnlabeledImages, Pix2PixDataset
from torchelie.datasets import SideBySideImagePairsDataset
from torchelie.models import *
from torchelie.recipes.algorithm import Algorithm


def get_dataset(dataset_specs: Tuple[str, str], img_size: int, train: bool):
    ty, path = dataset_specs
    if ty == 'pix2pix':
        return Pix2PixDataset('~/.torch',
                              path,
                              split='train' if train else 'val',
                              download=True,
                              transform=TF.Compose([
                                  TF.Resize(img_size),
                                  TF.RandomResizedCrop(img_size,
                                                       scale=(0.8, 1),
                                                       ratio=(1., 1.)),
                                  TF.RandomHorizontalFlip(),
                              ]))
    if ty == 'colorize':
        return UnlabeledImages(
            path,
            TF.Compose([
                TF.Resize(img_size),
                TF.RandomCrop(img_size),
                TF.RandomHorizontalFlip(),
                MultiBranch([
                    TF.Compose([
                        TF.Grayscale(3),
                        TF.ToTensor(),
                    ]),
                    TF.ToTensor(),
                ]),
            ]))
    if ty == 'inpainting':
        return UnlabeledImages(
            path,
            TF.Compose([
                TF.Resize(img_size),
                TF.RandomCrop(img_size),
                TF.RandomHorizontalFlip(),
                TF.ToTensor(),
                MultiBranch([
                    TF.Compose([
                        TF.RandomErasing(p=1, value=(1., 1., 0)),
                    ]),
                    TF.Compose([])
                ]),
            ]))
    if ty == 'edges':
        return UnlabeledImages(
            path,
            TF.Compose([
                TF.Resize(img_size),
                TF.CenterCrop(img_size),
                TF.RandomHorizontalFlip(),
                MultiBranch([
                    TF.Compose([
                        tch.transforms.Canny(),
                        TF.Grayscale(3),
                        TF.ToTensor(),
                    ]),
                    TF.Compose([
                        TF.ToTensor(),
                    ])
                ]),
            ]))
    if ty == 'pairs':
        return SideBySideImagePairsDataset(
            path,
            TF.Compose([
                TF.Resize(img_size),
                TF.CenterCrop(img_size),
                TF.RandomHorizontalFlip(),
                TF.ToTensor(),
            ]))
    assert False, "dataset's type not understood"


class GradientPenalty:

    @tu.experimental
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


@tu.experimental
class ConcatConditionalGANLoss:

    def __init__(self, G: nn.Module, D: nn.Module) -> None:
        self.G = G
        self.D = D
        self.gan_loss = tch.loss.gan.standard

        G_alg = Algorithm()

        @G_alg.add_step('adversarial')
        def G_adv_pass(env, src, dst):
            out = self.G(src * 2 - 1)
            with self.D.no_sync():
                loss = self.gan_loss.generated(
                    self.D(torch.cat([out, src], dim=1) * 2 - 1))
            env['loss'] = loss
            env['out'] = out
            return {'G_adv_loss': loss.item()}

        @G_alg.add_step('backward')
        def backward(env, src, dst):
            env['loss'].backward()
            return {}

        self.G_alg = G_alg

        D_alg = Algorithm()

        @D_alg.add_step('gen_fakes')
        def gen_fakes(env, src, dst):
            with self.G.no_sync():
                with torch.no_grad():
                    out = self.G(src * 2 - 1)
                env['out'] = out
                env['fake_pair'] = torch.cat([out, src], dim=1)
                env['real_pair'] = torch.cat([dst, src], dim=1)
            return {'out': env['out'].detach()}

        @D_alg.add_step('adversarial')
        def D_fake(env, src, dst):
            with D.no_sync():
                prob_fake = self.D(env['fake_pair'] * 2 - 1)
                fake_loss = self.gan_loss.fake(prob_fake)
                fake_loss.backward()

            prob_real = self.D(env['real_pair'] * 2 - 1)
            real_loss = self.gan_loss.real(prob_real)
            real_loss.backward()

            fake_correct = prob_fake.detach().lt(0).int().eq(1).sum().item()
            real_correct = prob_real.detach().gt(0).int().eq(1).sum()
            return {
                'fake_correct':
                    fake_correct,
                'real_correct':
                    real_correct,
                'fake_loss':
                    fake_loss.item(),
                'real_loss':
                    real_loss.item(),
                'prob_fake':
                    torch.sigmoid(prob_fake).mean().item(),
                'prob_real':
                    torch.sigmoid(prob_real).mean().item(),
                'D-correct':
                    (fake_correct + real_correct) / (2 * prob_fake.numel()),
            }

        self.D_alg = D_alg

    def G_step(self, src: torch.Tensor, dst: torch.Tensor) -> dict:
        return self.G_alg(src, dst)

    def D_step(self, src: torch.Tensor, dst: torch.Tensor) -> dict:
        return self.D_alg(src, dst)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + '\n' +
            tu.indent('G_steps:\n' + tu.indent(repr(self.G_alg)) +
                      "\n\nD_steps:\n" + tu.indent(repr(self.D_alg)) + "\n"))

    def use_r0(self, r0_gamma):
        self.gan_loss = tch.loss.gan.standard
        gradient_penalty = GradientPenalty(r0_gamma)

        @self.D_alg.insert_before('adversarial', 'D_r0')
        def D_r0(env, src, dst):
            with self.D.no_sync():
                g_norm = gradient_penalty(self.D, env['real_pair'] * 2 - 1,
                                          env['fake_pair'] * 2 - 1)
            return {'g_norm': g_norm}


@tu.experimental
class Pix2PixLoss(ConcatConditionalGANLoss):

    def __init__(self, G: nn.Module, D: nn.Module, l1_gain: float) -> None:
        super().__init__(G, D)
        self.l1_gain = l1_gain

        @self.G_alg.insert_after('adversarial', 'l1')
        def G_l1_pass(env, src, dst):
            loss = self.l1_gain * F.l1_loss(env['out'], dst)
            env['loss'] += loss
            return {'l1_loss': loss.item()}


@tu.experimental
class Pix2PixHDLoss(ConcatConditionalGANLoss):

    def __init__(self, G: nn.Module, D: nn.Module, l1_gain: float):
        super().__init__(G, D)
        self.gan_loss = tch.loss.gan.ls
        self.l1_gain = l1_gain

        D_with_acts = tnn.WithSavedActivations(D.module)

        @self.G_alg.insert_before('adversarial', 'extract_features')
        def G_features(env, src, dst):
            real_pair = torch.cat([dst, src], dim=1) * 2 - 1
            _, activations = D_with_acts(real_pair, detach=True)
            env['real_activations'] = activations

            fake = G(src * 2 - 1)
            fake_pair = torch.cat([fake, src], dim=1) * 2 - 1
            prob, activations = D_with_acts(fake_pair, detach=False)
            env['fake_activations'] = activations
            env['D_prob'] = prob
            env['fake'] = fake
            return {}

        @self.G_alg.insert_after('extract_features', 'feature_matching')
        def G_featmatch(env, src, dst):
            loss = 0.
            for key in env['real_activations'].keys():
                loss += F.l1_loss(env['fake_activations'][key],
                                  env['real_activations'][key])
            env['loss'] = self.l1_gain * loss / len(env['fake_activations'])
            return {'fm_loss': loss.item()}

        @self.G_alg.override_step('adversarial')
        def G_adv(env, src, dst):
            loss = self.gan_loss.generated(env['D_prob'])
            env['loss'] += loss
            return {'G_adv_loss': loss.item()}


@tu.experimental
def train(rank, world_size):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, type=lambda x: x.split(':'))
    parser.add_argument('--test-dataset',
                        required=True,
                        type=lambda x: x.split(':'))
    parser.add_argument('--r0-gamma', type=float)
    parser.add_argument('--G-type', choices=['hd', 'unet'], default='patch')
    parser.add_argument('--D-type',
                        choices=['patch', 'unet', 'multi'],
                        default='patch')
    parser.add_argument('--l1-gain', default=0, type=float)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--img-size', default=256, type=int)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--from-ckpt')
    opts = parser.parse_args()

    if opts.G_type == 'unet':
        G = pix2pix_256().to_instance_norm().to_equal_lr()
    else:
        G = pix2pixhd_dev().leaky().to_equal_lr()

    G_polyak = copy.deepcopy(G)
    if opts.D_type == 'patch':
        D = residual_patch286()
        D.set_input_specs(6)
        D.to_equal_lr()
        r0_gamma = 0.1
    elif opts.D_type == 'unet':
        D = UNet([32, 64, 128, 256, 512, 512], 1)
        D.set_decoder_num_layers(1)
        D.set_encoder_num_layers(1)
        D.set_input_specs(6)
        D.to_bilinear_sampling()
        D.remove_batchnorm()
        D.leaky()
        tnn.utils.net_to_equal_lr(D, leak=0.2)
        r0_gamma = 0.00001
    else:
        patch_D = patch70()
        patch_D.set_input_specs(6)
        patch_D.remove_batchnorm()
        tnn.utils.net_to_equal_lr(patch_DD)
        D = MultiScaleDiscriminator(patch_D)
        r0_gamma = 0.00001
    r0_gamma = opts.r0_gamma or r0_gamma

    if rank == 0:
        print(G)
        print(D)

    G = torch.nn.parallel.DistributedDataParallel(G.to(rank), [rank], rank)
    D = torch.nn.parallel.DistributedDataParallel(D.to(rank), [rank], rank)

    ds = get_dataset(opts.dataset, opts.img_size, train=True)
    ds = torch.utils.data.DataLoader(ds,
                                     opts.batch_size,
                                     num_workers=4,
                                     shuffle=True,
                                     pin_memory=True,
                                     drop_last=True)

    test_ds = get_dataset(opts.test_dataset, opts.img_size, train=False)
    test_ds = torch.utils.data.DataLoader(test_ds,
                                          opts.batch_size * 2,
                                          num_workers=4,
                                          shuffle=True,
                                          pin_memory=True,
                                          drop_last=True)

    pix2pix = Pix2PixHDLoss(G, D, opts.l1_gain)
    pix2pix.use_r0(opts.r0_gamma)
    print(pix2pix)

    def G_fun(batch) -> dict:
        x, y = batch
        if opts.reverse:
            x, y = y, x
        return pix2pix.G_step(x, y)

    def D_fun(batch) -> dict:
        x, y = batch
        if opts.reverse:
            x, y = y, x
        return pix2pix.D_step(x, y)

    def test_fun(batch):
        x, y = batch
        if opts.reverse:
            x, y = y, x
        G_polyak.train()
        out = G_polyak(x * 2 - 1).detach()
        return {'polyak_out': out, 'test_out': G(x * 2 - 1).detach()}

    tag = f'pix2pix_{opts.dataset[0]}:{os.path.basename(opts.dataset[1])}'
    recipe = GANRecipe(G,
                       D,
                       G_fun,
                       D_fun,
                       test_fun,
                       ds,
                       test_every=1000,
                       test_loader=test_ds,
                       checkpoint=tag if rank == 0 else None,
                       visdom_env=tag if rank == 0 else None)

    recipe.register('G_polyak', G_polyak)

    recipe.callbacks.add_callbacks([
        tch.callbacks.Optimizer(
            tch.optim.Lookahead(
                torch.optim.AdamW(D.parameters(),
                                  lr=2e-3,
                                  betas=(0., 0.99),
                                  weight_decay=0))),
        tch.callbacks.Log('out', 'out'),
        tch.callbacks.Log('batch.0', 'x'),
        tch.callbacks.Log('batch.1', 'y'),
        tch.callbacks.Log('fake_heatmap', 'fake_heatmap'),
        tch.callbacks.WindowedMetricAvg('fake_loss', 'fake_loss'),
        tch.callbacks.WindowedMetricAvg('real_loss', 'real_loss'),
        tch.callbacks.WindowedMetricAvg('prob_fake', 'prob_fake'),
        tch.callbacks.WindowedMetricAvg('prob_real', 'prob_real'),
        tch.callbacks.WindowedMetricAvg('D-correct', 'D-correct'),
        tch.callbacks.Log('g_norm', 'g_norm'),
        tch.callbacks.Throughput(),
    ])
    recipe.G_loop.callbacks.add_callbacks([
        tch.callbacks.Optimizer(
            tch.optim.Lookahead(
                torch.optim.AdamW(G.parameters(),
                                  lr=2e-3,
                                  betas=(0., 0.99),
                                  weight_decay=0))),
        tch.callbacks.Polyak(G.module, G_polyak),
    ])
    recipe.test_loop.callbacks.add_callbacks([
        tch.callbacks.Log('polyak_out', 'polyak_out'),
        tch.callbacks.Log('test_out', 'test_out'),
        tch.callbacks.Log('batch.0', 'test_x'),
        tch.callbacks.GANMetrics('batch.1', 'polyak_out', device=rank),
        tch.callbacks.Log('kid', 'kid'),
        tch.callbacks.Log('fid', 'fid'),
        tch.callbacks.Log('precision', 'precision'),
        tch.callbacks.Log('recall', 'recall'),
    ])
    recipe.to(rank)
    if opts.from_ckpt is not None:
        ckpt = torch.load(opts.from_ckpt, map_location='cpu')
        recipe.load_state_dict(ckpt)
    recipe.run(200)


if __name__ == '__main__':
    tu.parallel_run(train)
