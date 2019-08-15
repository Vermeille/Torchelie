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

    def __init__(self,
                 content_layer,
                 dream_layer,
                 ratio,
                 device='cpu',
                 **kwargs):
        super(DeepDreamRecipe, self).__init__(**kwargs)
        self.device = device
        self.loss = DeepDreamLoss(content_layer, dream_layer, ratio).to(device)

    def __call__(self, ref, nb_iters=500):
        canvas = ParameterizedImg(3, ref.height, ref.width).to(self.device)

        opt = torch.optim.Adam(canvas.parameters(),
                               lr=1e-1,
                               betas=(0.99, 0.999),
                               eps=1e-1)

        self.loss.set_content(TF.ToTensor()(ref).to(self.device))
        for iters in range(nb_iters):
            self.iters = iters
            opt.zero_grad()
            cim = canvas()
            loss = self.loss(cim)
            loss.backward()
            opt.step()

            self.log({'loss': loss, 'img': canvas.render()})

        return canvas.render()


if __name__ == '__main__':
    import argparse
    import torchvision.models as tvmodels

    parser = argparse.ArgumentParser(description="DeepDream")
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--ratio', type=float)
    parser.add_argument('--content-layer', default='conv4_2')
    parser.add_argument('--dream-layer', default='conv3_2')
    parser.add_argument('--visdom-env')
    args = parser.parse_args()

    dd = DeepDreamRecipe(args.content_layer,
                         args.dream_layer,
                         args.ratio,
                         device=args.device,
                         visdom_env=args.visdom_env)
    out = dd(Image.open(args.input), 4000)
    TF.ToPILImage()(out).save(args.out)
