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

import torch
import torchvision.transforms as TF

import torchelie.nn as tnn
import torchelie.metrics.callbacks as cb
from torchelie.data_learning import ParameterizedImg
from torchelie.loss.deepdreamloss import DeepDreamLoss
from torchelie.optim import DeepDreamOptim
from torchelie.recipes.recipebase import ImageOptimizationBaseRecipe

from PIL import Image


class DeepDreamRecipe(ImageOptimizationBaseRecipe):
    """
    Deep Dream recipe

    First instantiate the recipe then call `recipe(n_iter, img)`

    Args:
        model (nn.Module): the trained model to use
        dream_layer (str): the layer to use on which activations will be
            maximized
        lr (float, optional): the learning rate
        device (device): where to run the computation
        visdom_env (str or None): the name of the visdom env to use, or None
            to disable Visdom
    """
    def __init__(self,
                 model,
                 dream_layer,
                 lr=3e-4,
                 device='cpu',
                 visdom_env='deepdream'):
        super(DeepDreamRecipe, self).__init__(visdom_env=visdom_env)
        self.device = device
        self.loss = DeepDreamLoss(model, dream_layer).to(device)
        self.norm = tnn.ImageNetInputNorm().to(device)
        self.lr = lr

    def init(self, ref):
        ref_tensor = TF.ToTensor()(ref)
        self.canvas = ParameterizedImg(3,
                                       ref_tensor.shape[1],
                                       ref_tensor.shape[2],
                                       init_img=ref_tensor,
                                       space='spectral',
                                       colors='uncorr').to(self.device)

        self.opt = DeepDreamOptim(self.canvas.parameters(), lr=self.lr)

    def forward(self):
        self.opt.zero_grad()
        cim = self.canvas()
        loss = self.loss(
            self.norm(cim[:, :, self.iters % 10:, self.iters % 10:]))
        loss.backward()
        self.opt.step()
        return {'loss': loss}

    def result(self):
        return self.canvas.render()

    def __call__(self, n_iters, ref):
        """
        Run the recipe

        Args:
            n_iters (int): number of iterations to perform
            ref (PIL.Image): the base image

        Returns:
            the optimized image
        """
        return super(DeepDreamRecipe, self).__call__(n_iters, ref)


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
    dd = DeepDreamRecipe(model,
                         args.dream_layer or models[args.model]['layer'],
                         lr=args.lr,
                         device=args.device,
                         visdom_env=args.visdom_env)
    img = Image.open(args.input)
    out = dd(args.iters, img)

    TF.ToPILImage()(out).save(args.out)
