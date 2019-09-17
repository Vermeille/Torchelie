"""
Feature visualization

This optimizes an image to maximize some neuron in order to visualize the
features it captures

A commandline is provided with `python3 -m torchelie.recipes.feature_vis`
"""

import torch
import torchvision.transforms as TF

import torchelie
import torchelie.nn as tnn
import torchelie.metrics.callbacks as cb
from torchelie.optim import DeepDreamOptim
from torchelie.recipes.recipebase import ImageOptimizationBaseRecipe
from torchelie.data_learning import ParameterizedImg


class FeatureVisRecipe(ImageOptimizationBaseRecipe):
    """
    Feature viz

    First instantiate the recipe then call `recipe(n_iter, img)`

    Args:
        model (nn.Module): the trained model to use
        layer (str): the layer to use on which activations will be maximized
        input_size (int): the size of the square image the model accepts as
            input
        lr (float, optional): the learning rate
        device (device): where to run the computation
        visdom_env (str or None): the name of the visdom env to use, or None
            to disable Visdom
    """
    def __init__(self,
                 model,
                 layer,
                 input_size,
                 lr=1e-3,
                 device='cpu',
                 visdom_env='feature_vis'):
        super(FeatureVisRecipe, self).__init__(visdom_env=visdom_env)
        self.device = device
        self.model = tnn.WithSavedActivations(model, names=[layer]).to(device)
        self.model.eval()
        self.layer = layer
        self.input_size = input_size
        self.norm = tnn.ImageNetInputNorm().to(device)
        self.lr = lr

    def init(self, channel):
        self.channel = channel
        self.canvas = ParameterizedImg(3, self.input_size + 10,
                                       self.input_size + 10).to(self.device)

        self.opt = DeepDreamOptim(self.canvas.parameters(), lr=self.lr)

    def forward(self):
        self.opt.zero_grad()

        cim = self.canvas()
        im = cim[:, :, self.iters % 10:, self.iters % 10:]
        im = torch.nn.functional.interpolate(im,
                                             size=(self.input_size,
                                                   self.input_size),
                                             mode='bilinear')
        _, acts = self.model(self.norm(im), detach=False)
        fmap = acts[self.layer]
        loss = -fmap[0][self.channel].sum()
        loss.backward()
        self.opt.step()

        return {'loss': loss, 'x': self.canvas.render()}

    def result(self):
        return self.canvas.render()

    def __call__(self, n_iters, neuron):
        """
        Run the recipe

        Args:
            n_iters (int): number of iterations to run
            neuron (int): the the feature map to maximize

        Returns:
            the optimized image
        """
        return super(FeatureVisRecipe, self).__call__(n_iters, neuron)


if __name__ == '__main__':
    import argparse
    import torchvision.models as tvmodels

    models = {
        'vgg': {
            'ctor': tvmodels.vgg19,
            'sz': 224
        },
        'resnet': {
            'ctor': tvmodels.resnet18,
            'sz': 224
        },
        'googlenet': {
            'ctor': tvmodels.googlenet,
            'sz': 299
        },
        'inception': {
            'ctor': tvmodels.inception_v3,
            'sz': 299
        },
    }
    parser = argparse.ArgumentParser(description="Feature visualization")
    parser.add_argument('--model', required=True, choices=models.keys())
    parser.add_argument('--layer', required=True)
    parser.add_argument('--input-size')
    parser.add_argument('--neuron', required=True, type=int)
    parser.add_argument('--out', default='features.png')
    parser.add_argument('--visdom-env')
    args = parser.parse_args()

    choice = models[args.model]
    model = choice['ctor'](pretrained=True)
    print(model)
    fv = FeatureVisRecipe(model,
                          args.layer,
                          args.input_size or choice['sz'],
                          'cuda',
                          visdom_env=args.visdom_env)
    out = fv(4000, args.neuron)
    TF.ToPILImage()(out).save(args.out)
