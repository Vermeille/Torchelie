import torch
from torchvision.transforms import ToTensor, ToPILImage

from torchelie.loss import NeuralStyleLoss
from torchelie.data_learning import ParameterizedImg


def t2pil(t):
    return ToPILImage()(t)


def pil2t(pil):
    return ToTensor()(pil)


class NeuralStyleRecipe:
    def __init__(self, device="cpu"):
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

        prev_loss = None
        for i in range(100):

            def make_loss():
                opt.zero_grad()
                input_img = canvas()
                loss = self.loss(input_img)
                loss.backward()
                return loss

            loss = opt.step(make_loss).item()
            if prev_loss is not None and loss > prev_loss * 0.95:
                break
            prev_loss = loss

        return t2pil(canvas().detach().cpu()[0])
