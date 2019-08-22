import torch
import torchvision.transforms as TF

import torchelie
import torchelie.nn as tnn
import torchelie.metrics.callbacks as cb
from torchelie.optim import DeepDreamOptim
from torchelie.recipes.recipebase import ImageOptimizationBaseRecipe
from torchelie.data_learning import ParameterizedImg
"""
FIXME: Make a base class for image learning
"""


class FeatureVisRecipe(ImageOptimizationBaseRecipe):
    def __init__(self,
                 model,
                 layer,
                 input_size,
                 device='cpu',
                 visdom_env='feature_vis'):
        super(FeatureVisRecipe, self).__init__(visdom_env=visdom_env)
        self.device = device
        self.model = tnn.WithSavedActivations(model, names=[layer]).to(device)
        self.model.eval()
        self.layer = layer
        self.input_size = input_size
        self.norm = tnn.ImageNetInputNorm().to(device)

    def init(self, channel):
        self.channel = channel
        self.canvas = ParameterizedImg(3, self.input_size + 10,
                                       self.input_size + 10).to(self.device)

        self.opt = DeepDreamOptim(self.canvas.parameters(), lr=1e-3)

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
