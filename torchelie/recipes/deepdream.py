"""
Deep Dream recipe.

Performs the algorithm described in
https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html

This implementation differs from the original one: the image is optimized in
Fourier space, for greater details and colors, the model and layers are
customizable.

A commandline interface is provided through `python3 -m
torchelie.recipes.deepdream`, and a DeepDreamRecipe is provided.
"""
import random

import torch
import torchvision.transforms as TF

import torchelie.nn as tnn
from torchelie.data_learning import ParameterizedImg
from torchelie.loss.deepdreamloss import DeepDreamLoss
from torchelie.optim import DeepDreamOptim
from torchelie.recipes.recipebase import Recipe
import torchelie.callbacks as tcb

from PIL import Image


class DeepDream(torch.nn.Module):
    """
    Deep Dream recipe

    First instantiate the recipe then call `recipe(n_iter, img)`

    Args:
        model (nn.Module): the trained model to use
        dream_layer (str): the layer to use on which activations will be
            maximized
    """

    def __init__(self, model, dream_layer):
        super(DeepDream, self).__init__()
        self.loss = DeepDreamLoss(model, dream_layer)
        self.norm = tnn.ImageNetInputNorm()

    def fit(self, ref, iters, lr=3e-4, device='cpu', visdom_env='deepdream'):
        """
        Args:
            lr (float, optional): the learning rate
            visdom_env (str or None): the name of the visdom env to use, or None
                to disable Visdom
        """
        ref_tensor = TF.ToTensor()(ref).unsqueeze(0)
        canvas = ParameterizedImg(1, 3,
                                  ref_tensor.shape[2],
                                  ref_tensor.shape[3],
                                  init_img=ref_tensor,
                                  space='spectral',
                                  colors='uncorr')

        def forward(_):
            img = canvas()
            rnd = random.randint(0, 10)
            loss = self.loss(self.norm(img[:, :, rnd:, rnd:]))
            loss.backward()
            return {'loss': loss, 'img': img}

        loop = Recipe(forward, range(iters))
        loop.register('model', self)
        loop.register('canvas', canvas)
        loop.callbacks.add_callbacks([
            tcb.Counter(),
            tcb.Log('loss', 'loss'),
            tcb.Log('img', 'img'),
            tcb.Optimizer(DeepDreamOptim(canvas.parameters(), lr=lr)),
            tcb.VisdomLogger(visdom_env=visdom_env, log_every=10),
            tcb.StdoutLogger(log_every=10)
        ])
        loop.to(device)
        loop.run(1)
        return canvas.render().cpu()


if __name__ == '__main__':
    import argparse
    import torchvision.models as tvmodels

    models = {
        'vgg': {
            'ctor': tvmodels.vgg19,
            'layer': 'features.28'
        },
        'inception': {
            'ctor': tvmodels.inception_v3,
            'layer': 'Mixed_6c'
        },
        'googlenet': {
            'ctor': tvmodels.googlenet,
            'layer': 'inception4c'
        },
        'resnet': {
            'ctor': tvmodels.resnet18,
            'layer': 'layer3'
        }
    }
    parser = argparse.ArgumentParser(description="DeepDream")
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', default='googlenet', choices=models.keys())
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--iters', default=4000, type=int)
    parser.add_argument('--dream-layer')
    parser.add_argument('--visdom-env')
    args = parser.parse_args()

    model = models[args.model]['ctor'](pretrained=True)

    print(model)
    img = Image.open(args.input)
    dd = DeepDream(model, args.dream_layer or models[args.model]['layer'])
    out = dd.fit(img,
                 args.iters,
                 lr=args.lr,
                 device=args.device,
                 visdom_env=args.visdom_env)

    TF.ToPILImage()(out).save(args.out)
