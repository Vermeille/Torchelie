"""
Feature visualization

This optimizes an image to maximize some neuron in order to visualize the
features it captures

A commandline is provided with `python3 -m torchelie.recipes.feature_vis`
"""
import random

import torch
import torchvision.transforms as TF

import torchelie.nn as tnn
import torchelie.callbacks as tcb
from torchelie.optim import DeepDreamOptim
from torchelie.recipes.recipebase import Recipe
from torchelie.data_learning import ParameterizedImg


class FeatureVis(torch.nn.Module):
    """
    Feature viz

    First instantiate the recipe then call `recipe(n_iter, img)`

    Args:
        model (nn.Module): the trained model to use
        layer (str): the layer to use on which activations will be maximized
        input_size (int, or (int, int)): the size of the image the model
            accepts as input
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
        super().__init__()
        self.device = device
        self.model = tnn.WithSavedActivations(model, names=[layer])
        self.layer = layer
        if isinstance(input_size, (list, tuple)):
            self.input_size = input_size
        else:
            self.input_size = (input_size, input_size)
        self.norm = tnn.ImageNetInputNorm()
        self.lr = lr
        self.visdom_env = visdom_env

    def fit(self, n_iters, neuron):
        """
        Run the recipe

        Args:
            n_iters (int): number of iterations to run
            neuron (int): the feature map to maximize

        Returns:
            the optimized image
        """
        canvas = ParameterizedImg(1, 3,
                                  self.input_size[0] + 10,
                                  self.input_size[1] + 10)

        def forward(_):
            cim = canvas()
            rnd = random.randint(0, cim.shape[2] // 10)
            im = cim[:, :, rnd:, rnd:]
            im = torch.nn.functional.interpolate(im,
                                                 size=(self.input_size[0],
                                                       self.input_size[1]),
                                                 mode='bilinear')
            _, acts = self.model(self.norm(im), detach=False)
            fmap = acts[self.layer]
            loss = -fmap[0][neuron].sum()
            loss.backward()

            return {'loss': loss, 'img': cim}

        loop = Recipe(forward, range(n_iters))
        loop.register('canvas', canvas)
        loop.register('model', self)
        loop.callbacks.add_callbacks([
            tcb.Counter(),
            tcb.Log('loss', 'loss'),
            tcb.Log('img', 'img'),
            tcb.Optimizer(DeepDreamOptim(canvas.parameters(), lr=self.lr)),
            tcb.VisdomLogger(visdom_env=self.visdom_env, log_every=10),
            tcb.StdoutLogger(log_every=10)
        ])
        loop.to(self.device)
        loop.run(1)
        return canvas.render().cpu()


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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', default='features.png')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--iters', default=4000, type=int)
    parser.add_argument('--visdom-env')
    args = parser.parse_args()

    choice = models[args.model]
    model = choice['ctor'](pretrained=True)
    print(model)
    fv = FeatureVis(model,
                    args.layer,
                    args.input_size or choice['sz'],
                    lr=args.lr,
                    device=args.device,
                    visdom_env=args.visdom_env)
    out = fv.fit(args.iters, args.neuron)
    TF.ToPILImage()(out).save(args.out)
