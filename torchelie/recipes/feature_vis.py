import torch
import torchvision.transforms as TF

import torchelie
import torchelie.nn as tnn
from torchelie.recipes.recipebase import RecipeBase
from torchelie.data_learning import ParameterizedImg

"""
FIXME: Make a base class for image learning
FIXME: Make a customizable Optimizer from which we can derive Deep Dream's
optimizer. Use it for Deep Dream as well.
"""

class FeatureVisRecipe(RecipeBase):
    def __init__(self, model, layer, input_size, device='cpu', **kwargs):
        super(FeatureVisRecipe, self).__init__(**kwargs)
        self.device = device
        self.model = tnn.WithSavedActivations(model, names=[layer]).to(device)
        self.model.eval()
        self.layer = layer
        self.input_size = input_size
        self.norm = tnn.ImageNetInputNorm().to(device)

    def __call__(self, channel, nb_iters=500):
        canvas = ParameterizedImg(3,
                                  self.input_size + 10,
                                  self.input_size + 10).to(self.device)

        for iters in range(nb_iters):
            self.iters = iters
            def step():
                cim = canvas()
                im = cim[:, :, iters % 10:, iters % 10:]
                im = torch.nn.functional.interpolate(im,
                                                     size=(self.input_size,
                                                           self.input_size),
                                                     mode='bilinear')
                _, acts = self.model(self.norm(im), detach=False)
                fmap = acts[self.layer]
                loss = -fmap[0, channel].sum()
                loss.backward()
                return loss
            loss = step()
            for p in canvas.parameters():
                p.data -= 1e-3 * p.grad.data / p.grad.abs().mean()
                p.grad.data.zero_()

            self.log({'loss': loss, 'img': canvas.render()})

        return canvas.render()


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
    out = fv(args.neuron, 4000)
    TF.ToPILImage()(out).save(args.out)
