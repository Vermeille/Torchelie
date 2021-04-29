"""
Neural Style from Leon Gatys

A commandline interface is provided through `python3 -m
torchelie.recipes.neural_style`
"""

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

import torchelie as tch
from torchelie.loss import NeuralStyleLoss
from torchelie.data_learning import ParameterizedImg
from torchelie.recipes.recipebase import Recipe
import torchelie.callbacks as tcb


class NeuralStyle(torch.nn.Module):
    """
    Neural Style Recipe

    First instantiate the recipe then call `recipe(n_iter, img)`

    Args:
        lr (float, optional): the learning rate
        device (device): where to run the computation
        visdom_env (str or None): the name of the visdom env to use, or None
            to disable Visdom
    """

    def __init__(self, lr=0.01, device="cpu", visdom_env='style'):
        super(NeuralStyle, self).__init__()
        self.loss = NeuralStyleLoss()
        self.loss2 = NeuralStyleLoss()
        self.device = device
        self.lr = lr
        self.visdom_env = visdom_env

    def fit(self,
            iters,
            content_img,
            style_img,
            style_ratio,
            *,
            second_scale_ratio=1,
            content_layers=None,
            init_with_content=False):
        """
        Run the recipe

        Args:
            n_iters (int): number of iterations to run
            content (PIL.Image): content image
            style (PIL.Image): style image
            ratio (float): weight of style loss
            content_layers (list of str): layers on which to reconstruct
                content
        """
        self.loss.to(self.device)
        self.loss.set_style(to_tensor(style_img).to(self.device), style_ratio)
        self.loss.set_content(
            to_tensor(content_img).to(self.device), content_layers)

        self.loss2.to(self.device)
        self.loss2.set_style(
            torch.nn.functional.interpolate(to_tensor(style_img)[None],
                                            scale_factor=0.5,
                                            mode='bilinear',
                                            align_corners=False,
                                            recompute_scale_factor=True)[0].to(
                                                self.device), style_ratio)
        self.loss2.set_content(
            torch.nn.functional.interpolate(to_tensor(content_img)[None],
                                            scale_factor=0.5,
                                            mode='bilinear',
                                            align_corners=False,
                                            recompute_scale_factor=True)[0].to(
                                                self.device), content_layers)

        canvas = ParameterizedImg(1,
                                  3,
                                  content_img.height,
                                  content_img.width,
                                  init_img=to_tensor(content_img).unsqueeze(0)
                                  if init_with_content else None)

        self.opt = tch.optim.RAdamW(canvas.parameters(),
                                    3e-2,
                                    betas=(0.9, 0.99),
                                    weight_decay=0)

        def forward(_):
            img = canvas()
            loss, losses = self.loss(img)
            loss.backward()
            loss, losses = self.loss2(
                torch.nn.functional.interpolate(canvas(),
                                                scale_factor=0.5,
                                                mode='bilinear',
                                                align_corners=False,
                                                recompute_scale_factor=True))
            (second_scale_ratio * loss).backward()

            return {
                'loss': loss,
                'content_loss': losses['content_loss'],
                'style_loss': losses['style_loss'],
                'hists_loss': losses['hists_loss'],
                'img': img,
            }

        loop = Recipe(forward, range(iters))
        loop.register('canvas', canvas)
        loop.register('model', self)
        loop.callbacks.add_callbacks([
            tcb.Counter(),
            tcb.WindowedMetricAvg('loss'),
            tcb.WindowedMetricAvg('content_loss'),
            tcb.WindowedMetricAvg('style_loss'),
            tcb.WindowedMetricAvg('hists_loss'),
            tcb.Log('img', 'img'),
            tcb.VisdomLogger(visdom_env=self.visdom_env, log_every=10),
            tcb.StdoutLogger(log_every=10),
            tcb.Optimizer(self.opt, log_lr=True),
        ])
        loop.to(self.device)
        loop.run(1)
        return canvas.render().cpu()


if __name__ == '__main__':
    import argparse
    import sys
    from PIL import Image
    parser = argparse.ArgumentParser(
        description="Implementation of Neural Artistic Style by Gatys")
    parser.add_argument('--content', required=True)
    parser.add_argument('--style', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--size', type=int)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--ratio', default=1, type=float)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--content-layers',
                        default=None,
                        type=lambda x: x and x.split(','))
    parser.add_argument('--iters', default=100, type=int)
    parser.add_argument('--visdom-env')
    args = parser.parse_args(sys.argv[1:])

    stylizer = NeuralStyle(device=args.device, visdom_env=args.visdom_env)

    content = Image.open(args.content)
    content.thumbnail((args.size, args.size))

    style_img = Image.open(args.style)
    if args.scale != 1.0:
        new_style_size = (int(style_img.width * args.scale),
                          int(style_img.height * args.scale))
        style_img = style_img.resize(new_style_size, Image.BICUBIC)

    result = stylizer.fit(args.iters, content, style_img, args.ratio,
                          args.content_layers)
    result = to_pil_image(result)

    result.save(args.out)
