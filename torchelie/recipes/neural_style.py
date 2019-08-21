import torch
from torchvision.transforms import ToTensor, ToPILImage

from torchelie.loss import NeuralStyleLoss
from torchelie.data_learning import ParameterizedImg
from torchelie.recipes.recipebase import ImageOptimizationBaseRecipe
import torchelie.metrics.callbacks as cb


def t2pil(t):
    return ToPILImage()(t)


def pil2t(pil):
    return ToTensor()(pil)


class NeuralStyleRecipe(ImageOptimizationBaseRecipe):
    def __init__(self, device="cpu", visdom_env='style'):
        super(NeuralStyleRecipe, self).__init__(callbacks=[
            cb.WindowedLossAvg(),
            cb.LogInput(),
            cb.VisdomLogger(visdom_env, log_every=1),
            cb.StdoutLogger(log_every=1),
        ])

        self.loss = NeuralStyleLoss().to(device)
        self.device = device

    def init(self, content_img, style_img, style_ratio, content_layers=None):
        self.loss.set_style(pil2t(style_img).to(self.device), style_ratio)
        self.loss.set_content(
            pil2t(content_img).to(self.device), content_layers)

        self.canvas = ParameterizedImg(3, content_img.height,
                                       content_img.width).to(self.device)

        self.opt = torch.optim.LBFGS(self.canvas.parameters(),
                                     lr=0.01,
                                     history_size=50)

    def forward(self):
        def make_loss():
            self.opt.zero_grad()
            input_img = self.canvas()
            loss = self.loss(input_img)
            loss.backward()
            return loss

        loss = self.opt.step(make_loss).item()

        return {'loss': loss, 'x': self.canvas.render()}

    def result(self):
        return t2pil(self.canvas.render())


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
    parser.add_argument('--visdom-env')
    args = parser.parse_args(sys.argv[1:])

    stylizer = NeuralStyleRecipe(device=args.device,
                                 visdom_env=args.visdom_env)

    content = Image.open(args.content)
    content.thumbnail((args.size, args.size))

    style_img = Image.open(args.style)
    if args.scale != 1.0:
        new_style_size = (int(style_img.width * args.scale),
                          int(style_img.height * args.scale))
        style_img = style_img.resize(new_style_size, Image.BICUBIC)

    result = stylizer(100, content, style_img, args.ratio, args.content_layers)

    result.save(args.out)
