import torch
from torchelie.models import *


def test_patchgan():
    for M in [patch286, patch70, patch34, patch16]:
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
    m = ResidualDiscriminator([2, 'D', 3])
    m(torch.randn(1, 3, 8, 8))

    m = ResidualDiscriminator([2, 'D', 3])
    m.to_projection_discr(3)
    m(torch.randn(1, 3, 8, 8), torch.LongTensor([1]))

    def run(M):
        m = M(4)
        out = m(torch.randn(2, 3, 32, 32))
        out.mean().backward()

    run(resnet18)
    run(preact_resnet18)
    run(resnet50)
    run(preact_resnet50)
    run(resnext50_32x4d)
    run(preact_resnext50_32x4d)


def test_unet():
    m = UNet([3, 6, 12], 1)
    m(torch.randn(1, 3, 128, 128))


def test_vgg():
    m = vgg11(2)
    m(torch.randn(1, 3, 32, 32))


def test_attention():
    m = attention56(2)
    m(torch.randn(2, 3, 32, 32))


def test_hourglass():
    m = Hourglass()
    out = m(torch.randn(2, 32, 128, 128))
    assert out.shape == (2, 3, 128, 128)


def test_autogan():
    m = AutoGAN([3, 4, 5], in_noise=4)
    m(torch.randn(16, 4))
