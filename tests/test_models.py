import torch
from torchelie.models import *


def test_patchgan():
    """
    Test for patchgan.

    Args:
    """
    for M in [Patch286, Patch70, Patch32, Patch16]:
        m = M()
        m(torch.randn(1, 3, 128, 128))

    for M in [ProjPatch32]:
        m = M()
        m(torch.randn(1, 3, 128, 128), torch.LongTensor([0]))


def test_pnet():
    """
    Test if the pnet

    Args:
    """
    pnet = PerceptualNet(['conv5_2'])
    pnet(torch.randn(1, 3, 128, 128), detach=True)


def test_factored_predictor():
    """
    Predict the factored factors.

    Args:
    """
    fp = PixelPredictor(10)
    fp(torch.randn(5, 10), torch.randn(5, 3))
    fp.sample(torch.randn(3, 10), 1)


def test_pixelcnn():
    """
    Test if the pixel pixel pixel is a 2 - pixel pixel pixel pixel.

    Args:
    """
    pc = PixelCNN(10, (8, 8), 1)
    pc(torch.randn(1, 1, 8, 8))
    pc.sample(1, 1)

    pc = PixelCNN(10, (8, 8), 3)
    pc(torch.randn(1, 3, 8, 8))
    pc.sample(1, 1)


def test_resnet():
    """
    Test for resnet resnet.

    Args:
    """
    m = ResNetDebug(3)
    m(torch.randn(1, 3, 32, 32))

    m = PreactResNetDebug(3)
    m(torch.randn(1, 3, 32, 32))

    m = ClassCondResNetDebug(3, 2)
    m(torch.randn(1, 3, 32, 32), torch.LongTensor([1]))

    m = VectorCondResNetDebug(12)
    m(torch.randn(1, 3, 32, 32), torch.randn(1, 12))

    m = snres_discr_ctor([2, 'D', 3], in_ch=3)
    m(torch.randn(1, 3, 8, 8))

    m = snres_projdiscr([2, 'D', 3], in_ch=3, num_classes=4)
    m(torch.randn(1, 3, 8, 8), torch.LongTensor([1]))

    resnet18(1)(torch.randn(1, 3, 32, 32))
    preact_resnet18(1)(torch.randn(1, 3, 64, 64))
    preact_resnet34(1)(torch.randn(1, 3, 64, 64))
    preact_resnet20_cifar(1)(torch.randn(1, 3, 64, 64))
    resnet20_cifar(1)(torch.randn(1, 3, 32, 32))


def test_unet():
    """
    Test if a test.

    Args:
    """
    m = UNet()
    m(torch.randn(1, 3, 128, 128))


def test_vgg():
    """
    Generate a random vgg.

    Args:
    """
    m = VggDebug(2)
    m(torch.randn(1, 1, 32, 32))

    m = VggGeneratorDebug()
    m(torch.randn(1, 32))

    m = VggImg2ImgGeneratorDebug(8, 1, 1)
    m(torch.randn(1, 8), torch.randn(1, 1, 32, 32))

    m = VggClassCondGeneratorDebug(8, 1, 1)
    m(torch.randn(1, 8), torch.LongTensor([0]))


def test_attention():
    """
    Generate a random attention.

    Args:
    """
    m = attention56(2)
    m(torch.randn(2, 3, 32, 32))

def test_hourglass():
    """
    Test if the test test test test.

    Args:
    """
    m = Hourglass()
    m(torch.randn(2, 32, 128, 128))


def test_autogan():
    """
    Autogan noise.

    Args:
    """
    m = AutoGAN([3, 4, 5], in_noise=4)
    m(torch.randn(16, 4))
