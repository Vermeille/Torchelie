import torch
import torchvision.transforms as TF

import torchelie.nn as tnn
from torchelie.data_learning import ParameterizedImg


class FeatureVisRecipe:
    def __init__(self, model, layer, input_size, device='cpu'):
        self.device = device
        self.model = tnn.WithSavedActivations(model, names=[layer]).to(device)
        self.model.eval()
        self.layer = layer
        self.input_size = input_size
        self.norm = tnn.ImageNetInputNorm().to(device)

    def __call__(self, channel, nb_iters=500):
        canvas = ParameterizedImg(3, self.input_size + 10,
                                  self.input_size + 10).to(self.device)

        opt = torch.optim.Adam(canvas.parameters(),
                               lr=1e-1,
                               betas=(0.99, 0.999),
                               eps=1e-1)

        for iters in range(nb_iters):
            opt.zero_grad()
            cim = canvas()
            im = cim[:, :, iters % 10:, iters % 10:]
            im = torch.nn.functional.interpolate(im,
                                                 size=(self.input_size,
                                                       self.input_size),
                                                 mode='bilinear')
            _, acts = self.model(self.norm(im), detach=False)
            fmap = acts[self.layer]
            loss = -fmap[0, channel].mean()
            loss.backward()
            opt.step()

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
            'ctor': tvmodels.resnet101,
            'sz': 224
        },
        'inception': {
            'ctor': tvmodels.inception_v3,
            'sz': 299
        },
    }
    parser = argparse.ArgumentParser(description="Feature visualization")
    parser.add_argument('--model', required=True, choices=models.keys())
    parser.add_argument('--layer', required=True)
    parser.add_argument('--neuron', required=True, type=int)
    parser.add_argument('--out', default='features.png')
    args = parser.parse_args()

    choice = models[args.model]
    model = choice['ctor'](pretrained=True)
    print(model)
    fv = FeatureVisRecipe(model, args.layer, choice['sz'], 'cuda')
    out = fv(args.neuron, 4000)
    TF.ToPILImage()(out).save(args.out)
