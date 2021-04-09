import torch
from torchelie.models import *


def test_patchgan():
    for M in [Patch286, Patch70, Patch32, Patch16]:
        m = M()
        m(torch.randn(1, 3, 128, 128))


def test_pnet():
    pnet = PerceptualNet(['conv5_2'])
    pnet(torch.randn(1, 3, 128, 128), detach=True)


def test_factored_predictor():
    fp = PixelPredictor(10)
    fp(torch.randn(5, 10), torch.randn(5, 3))
    fp.sample(torch.randn(3, 10), 1)


def test_pixelcnn():
    pc = PixelCNN(10, (8, 8), 1)
    pc(torch.randn(1, 1, 8, 8))
    pc.sample(1, 1)

    pc = PixelCNN(10, (8, 8), 3)
    pc(torch.randn(1, 3, 8, 8))
    pc.sample(1, 1)


def test_resnet():
    m = snres_discr_ctor([2, 'D', 3], in_ch=3)
    m(torch.randn(1, 3, 8, 8))

    m = snres_projdiscr([2, 'D', 3], in_ch=3, num_classes=4)
    m(torch.randn(1, 3, 8, 8), torch.LongTensor([1]))

    for M in [resnet18, preact_resnet18, resnet50, preact_resnet50, resnext50,
            preact_resnext50]:
        m = M(4)
        out = m(torch.randn(2, 3, 32, 32))
        out.mean().backward()


def test_unet():
    m = UNet()
    m(torch.randn(1, 3, 128, 128))


def test_vgg():
    m = VggDebug(2)
    m(torch.randn(1, 1, 32, 32))


def test_attention():
    m = attention56(2)
    m(torch.randn(2, 3, 32, 32))

def test_hourglass():
    m = Hourglass()
    m(torch.randn(2, 32, 128, 128))


def test_autogan():
    m = AutoGAN([3, 4, 5], in_noise=4)
    m(torch.randn(16, 4))
