import torch
import torchvision.transforms as TF

import torchelie.nn as tnn
from torchelie.recipes.recipebase import RecipeBase
from torchelie.data_learning import ParameterizedImg
from torchelie.loss.deepdreamloss import DeepDreamLoss

from PIL import Image


class DeepDreamRecipe(RecipeBase):
    """
    FIXME: this deep dream implementation differs significantly from the
    official one:
        - it doesn't use tiling (but does octaves)
        - it uses standard backprop with Adam instead of DeepDream's SGD +
          gradient scaling
        - it does not use the pictue as an initial state but as a content loss,
          similar to artistic style.
        - it uses vgg instead of inception
    However, this is good enough for now.
    """

    def __init__(self, model, dream_layer, device='cpu', **kwargs):
        super(DeepDreamRecipe, self).__init__(**kwargs)
        self.device = device
        self.loss = DeepDreamLoss(model, dream_layer).to(device)
        self.norm = tnn.ImageNetInputNorm().to(device)

    def __call__(self, ref, nb_iters=500):
        ref_tensor = TF.ToTensor()(ref)
        canvas = ParameterizedImg(3,
                                  ref_tensor.shape[1],
                                  ref_tensor.shape[2],
                                  init_img=ref_tensor,
                                  space='spectral',
                                  colors='uncorr').to(self.device)

        for iters in range(nb_iters):
            self.iters = iters
            cim = canvas()
            loss = self.loss(self.norm(cim[:, :, iters % 10:, iters % 10:]))
            loss.backward()
            for p in canvas.parameters():
                p.data -= 3e-4 * p.grad.data / p.grad.abs().mean()
                p.grad.data.zero_()

            self.log({'loss': loss, 'img': canvas.render()})

        return canvas.render()


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
    parser.add_argument('--dream-layer')
    parser.add_argument('--visdom-env')
    args = parser.parse_args()

    model = models[args.model]['ctor'](pretrained=True)

    print(model)
    dd = DeepDreamRecipe(model,
                         args.dream_layer or models[args.model]['layer'],
                         device=args.device,
                         visdom_env=args.visdom_env)
    img = Image.open(args.input)
    out = dd(img, 4000)

    TF.ToPILImage()(out).save(args.out)
