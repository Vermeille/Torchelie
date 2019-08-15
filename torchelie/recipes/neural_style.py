import torch
from torchvision.transforms import ToTensor, ToPILImage

from torchelie.loss import NeuralStyleLoss
from torchelie.data_learning import ParameterizedImg
from torchelie.recipes.recipebase import RecipeBase


def t2pil(t):
    return ToPILImage()(t)


def pil2t(pil):
    return ToTensor()(pil)


class NeuralStyleRecipe(RecipeBase):
    def __init__(self, device="cpu", **kwargs):
        super(NeuralStyleRecipe, self).__init__(log_every=1, **kwargs)
        self.loss = NeuralStyleLoss().to(device)
        self.device = device

    def build_ref_acts(self, content_img, style_img, style_ratio,
                       content_layers):
        self.loss.set_style(style_img.to(self.device), style_ratio)
        self.loss.set_content(content_img.to(self.device), content_layers)

    def __call__(self,
                 content_img,
                 style_img,
                 style_ratio,
                 content_layers=None):
        self.build_ref_acts(pil2t(content_img), pil2t(style_img), style_ratio,
                            content_layers)
        canvas = ParameterizedImg(3, content_img.height,
                                  content_img.width).to(self.device)
        return self.optimize_img(canvas)

    def optimize_img(self, canvas):
        opt = torch.optim.LBFGS(canvas.parameters(), lr=0.01, history_size=50)

        self.iters = 0
        for i in range(100):
            def make_loss():
                opt.zero_grad()
                input_img = canvas()
                loss = self.loss(input_img)
                loss.backward()
                return loss

            loss = opt.step(make_loss).item()

            self.log({'loss': loss, 'img': canvas.render()})
            self.iters += 1

        return t2pil(canvas.render())

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
    parser.add_argument('--content-layers', default=None,
            type=lambda x: x and x.split(','))
    parser.add_argument('--visdom-env')
    args = parser.parse_args(sys.argv[1:])

    stylizer = NeuralStyleRecipe(device=args.device, visdom_env=args.visdom_env)

    content = Image.open(args.content)
    content.thumbnail((args.size, args.size))

    style_img = Image.open(args.style)
    if args.scale != 1.0:
        new_style_size = (
                int(style_img.width * args.scale),
                int(style_img.height * args.scale)
        )
        style_img = style_img.resize(new_style_size, Image.BICUBIC)

    result = stylizer(content, style_img, args.ratio, args.content_layers)

    result.save(args.out)
