"""
Deep Image Prior as a command line tool is in this recipe, it provides
super-resolution and denoising.

:code:`python3 -m torchelie.recipes.image_prior` for the usage message
"""
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TFF
from torchelie.models import Hourglass
import torchelie.transforms as ttf
import torchelie as tch
from torchelie.recipes.trainandcall import TrainAndCall
import torchelie.callbacks as tcb


def with_patches(img, patch_size, task, *args, **kwargs):
    if img.width < patch_size and img.height < patch_size:
        return task(img, *args, **kwargs)

    transformed = [(coords, task(im, *args, **kwargs))
                   for coords, im in ttf.patches(img, patch_size)]

    return ttf.paste_patches(transformed)


def with_patches2(img, mask, patch_size, task, *args, **kwargs):
    assert img.size == mask.size, 'Sizes of mask and image do not match'

    if img.width < patch_size and img.height < patch_size:
        return task(img, mask, *args, **kwargs)

    transformed = [(coords, task(im, ma, *args, **kwargs))
                   for (coords,
                        im), (_, ma) in zip(ttf.patches(img, patch_size),
                                            ttf.patches(mask, patch_size))]

    return ttf.paste_patches(transformed)


def input_noise(size, channels):
    if channels != 2:
        lines = torch.linspace(-0.01, 0.01, size[0])
        cols = torch.linspace(-0.01, 0.01, size[1])
        ll, cc = torch.meshgrid(lines, cols)
        ll = ll.unsqueeze(0).unsqueeze(0)
        cc = cc.unsqueeze(0).unsqueeze(0)
        return cc + ll + torch.rand(1, channels, size[0], size[1]) / 10 - 0.05
    else:
        lines = torch.linspace(-4, 4, size[1])
        cols = torch.linspace(-4, 4, size[0])
        ll, cc = torch.meshgrid(lines, cols)
        ll = ll.unsqueeze(0).unsqueeze(0)
        cc = cc.unsqueeze(0).unsqueeze(0)
        return torch.cat([ll, cc], dim=1)


def inpainting(img,
               mask,
               hourglass,
               input_dim,
               iters,
               lr,
               noise_std=1 / 30,
               device='cuda'):
    im = TFF.to_tensor(img)[None].to(device)
    mask = TFF.to_tensor(mask)[None].to(device)
    z = input_noise((im.shape[2], im.shape[3]), input_dim)
    z = z.to(device)
    print(hourglass)

    def body(batch):
        recon = hourglass(z + torch.randn_like(z) * noise_std)
        loss = torch.sum(
            F.mse_loss(F.interpolate(recon, size=im.shape[2:], mode='nearest'),
                       im,
                       reduction='none') * mask / mask.sum())
        loss.backward()
        return {"loss": loss}

    def display():
        recon = hourglass(z)
        recon = F.interpolate(recon, size=im.shape[2:], mode='nearest')
        loss = F.mse_loss(recon * mask, im)

        result = recon * (1 - mask) + im * mask
        return {
            "loss": loss,
            "recon": recon.clamp(0, 1),
            'orig': im,
            'result': result.clamp(0, 1)
        }

    loop = make_loop(hourglass, body, display, iters, lr)
    loop.test_loop.callbacks.add_callbacks([tcb.Log('result', 'result')])
    loop.to(device)
    loop.run(1)
    with torch.no_grad():
        hourglass.eval()
        return TFF.to_pil_image(hourglass(z)[0].cpu())


def superres(img,
             hourglass,
             input_dim,
             scale,
             iters,
             lr,
             noise_std=1 / 30,
             device='cuda'):
    im = TFF.to_tensor(img)[None].to(device)
    z = input_noise((im.shape[2] * scale, im.shape[3] * scale), input_dim)
    z = z.to(device)

    def body(batch):
        recon = hourglass(z + torch.randn_like(z) * noise_std)
        loss = F.mse_loss(
            F.interpolate(recon, size=im.shape[2:], mode='bilinear'), im)
        loss.backward()
        return {
            "loss": loss,
        }

    def display():
        recon = hourglass(z)
        loss = F.mse_loss(
            F.interpolate(recon, size=im.shape[2:], mode='bilinear'), im)
        return {
            "loss":
            loss,
            "recon":
            recon.clamp(0, 1),
            'orig':
            F.interpolate(im, scale_factor=scale, mode='bicubic').clamp(0, 1)
        }

    loop = make_loop(hourglass, body, display, iters, lr)
    loop.to(device)
    loop.run(1)
    with torch.no_grad():
        hourglass.eval()
        return TFF.to_pil_image(hourglass(z)[0].cpu())


def make_loop(hourglass, body, display, num_iter, lr):
    loop = TrainAndCall(hourglass,
                        body,
                        display,
                        range(num_iter),
                        test_every=50,
                        checkpoint=None)
    opt = tch.optim.RAdamW(hourglass.parameters(), lr=lr)
    loop.callbacks.add_callbacks([
        tcb.WindowedMetricAvg('loss'),
        tcb.Optimizer(opt, clip_grad_norm=0.5, log_lr=True),
    ])
    loop.test_loop.callbacks.add_callbacks([
        tcb.Log('recon', 'img'),
        tcb.Log('orig', 'orig'),
        tcb.Log('loss', 'loss'),
    ])
    return loop


if __name__ == '__main__':
    import argparse

    hyper_params = {
        '2xsuperres': {
            'fun': 'superres',
            'noise_std': 1 / 30,
            'lr': 0.01,
            'iters': 2000,
            'model': 'superres',
            'scale': 2
        },
        '4xsuperres': {
            'fun': 'superres',
            'noise_std': 1 / 30,
            'lr': 0.01,
            'iters': 2000,
            'model': 'superres',
            'scale': 4
        },
        '8xsuperres': {
            'fun': 'superres',
            'noise_std': 1 / 20,
            'lr': 0.01,
            'iters': 4000,
            'model': 'superres',
            'scale': 8
        },
        'text-inpainting': {
            'fun': 'inpainting',
            'noise_std': 1 / 30,
            'lr': 0.01,
            'iters': 6000,
            'model': 'superres'
        },
        'hole-inpainting-1': {
            'fun': 'inpainting',
            'noise_std': 1 / 30,
            'lr': 0.01,
            'iters': 6000,
            'model': 'inpainting-1'
        },
        'hole-inpainting-2': {
            'fun': 'inpainting',
            'noise_std': 0,
            'lr': 0.1,
            'iters': 5000,
            'model': 'inpainting-2'
        },
        'denoising': {
            'fun': 'superres',
            'noise_std': 1 / 30,
            'lr': 0.01,
            'iters': 1800,
            'model': 'superres',
            'scale': 1
        },
        'jpeg-removal': {
            'fun': 'superres',
            'noise_std': 1 / 30,
            'lr': 0.01,
            'iters': 2400,
            'model': 'inpainting-2',
            'scale': 1
        }
    }
    models = {
        'superres': {
            'noise_dim': 32,
            'down_channels': [128] * 6,
            'down_kernel': [3] * 6,
            'up_kernel': [3] * 6,
            'skip_channels': 4,
            'upsampling': 'bilinear'
        },
        'inpainting-1': {
            'noise_dim': 2,
            'down_channels': [128] * 6,
            'down_kernel': [3] * 6,
            'up_kernel': [3] * 6,
            'skip_channels': 0,
            'upsampling': 'bilinear'
        },
        'inpainting-2': {
            'noise_dim': 32,
            'down_channels': [16, 32, 64, 128, 128, 128],
            'down_kernel': [3] * 6,
            'up_kernel': [5] * 6,
            'skip_channels': 0,
            'upsampling': 'nearest'
        }
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--task',
                        required=True,
                        choices=hyper_params.keys(),
                        type=str)
    parser.add_argument('--mask', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--noise-std', type=float)
    parser.add_argument('--output', default='out.jpg', type=str)
    parser.add_argument('--n-layers', default=6, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-size', type=int)
    parser.add_argument('--patch-size', default=1024, type=int)
    opts = parser.parse_args()

    im = Image.open(opts.input).convert('RGB')
    if opts.max_size is not None:
        im.thumbnail((opts.max_size, opts.max_size))

    hps = hyper_params[opts.task]
    for k, v in hps.items():
        hps[k] = getattr(opts, k, v) or v

    model_params = models[hps.pop('model')]
    for k, v in model_params.items():
        if isinstance(v, list):
            model_params[k] = v[:opts.n_layers]

    model = Hourglass(**model_params)

    fun_name = hps.pop('fun')
    fun = globals()[fun_name]
    print(fun, hps)

    if 'inpainting' in fun_name:
        if opts.mask is not None:
            mask = Image.open(opts.mask).convert('RGB')
            if opts.max_size is not None:
                mask.thumbnail((opts.max_size, opts.max_size))

        out = with_patches2(im, mask, opts.patch_size, inpainting, model,
                            model_params['noise_dim'], **hps)
    else:
        out = with_patches(im, opts.patch_size, fun, model,
                           model_params['noise_dim'], **hps)

    out.save(opts.output)
