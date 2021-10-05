torchelie.models
================

We provide trained models. Use the argument :code:`pretrained=task` in order to
use them. Example: :code:`torchelie.models.resnet18(1000, pretrained='classification/imagenet')`.

Alternatively: :code:`torchelie.models.get_model('resnet18', 1000, pretrained='classification/imagenet')`

.. list-table:: Pretrained models
   :header-rows: 1

   * - model
     - task
     - notes
     - source

   * - resnet18
     - classification/imagenet
     - top1: 69.75%
     - torchvision

   * - resnet34
     - classification/imagenet
     - top1: 73.31%
     - torchvision

   * - resnet50
     - classification/imagenet
     - top1: 76.13%
     - torchvision

   * - resnet101
     - classification/imagenet
     - top1: 77.37%
     - torchvision

   * - resnet152
     - classification/imagenet
     - top1: 78.31%
     - torchvision

   * - **preact_resnet18**
     - **classification/imagenet**
     - top1: 68.41% (192x192 crop)
     - **torchelie**

   * - vgg11
     - classification/imagenet
     - top1: 69.02%
     - torchvision

   * - vgg13
     - classification/imagenet
     - top1: 69.92%
     - torchvision

   * - vgg16
     - classification/imagenet
     - top1: 71.59%
     - torchvision

   * - vgg19
     - classification/imagenet
     - top1: 72.37%
     - torchvision

   * - vgg11_bn
     - classification/imagenet
     - top1: 70.37%
     - torchvision

   * - vgg13_bn
     - classification/imagenet
     - top1: 71.58%
     - torchvision

   * - vgg16_bn
     - classification/imagenet
     - top1: 73.36%
     - torchvision

   * - vgg19_bn
     - classification/imagenet
     - top1: 74.21%
     - torchvision

   * - **vgg19**
     - **perceptual/imagenet**
     - activations normalized to have mean 1 for perceptual losses
     - **torchelie**

.. currentmodule:: torchelie.models

VGG
~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   VGG
   vgg11
   vgg13
   vgg16
   vgg19
   vgg11_bn
   vgg13_bn
   vgg16_bn
   vgg19_bn

Resnets
~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   ResNet
   ResNetInput
   preact_resnet101
   preact_resnet152
   preact_resnet18
   preact_resnet20_cifar
   preact_resnet34
   preact_resnet50
   preact_resnext101_32x4d
   preact_resnext152_32x4d
   preact_resnext50_32x4d
   preact_wide_resnet101
   preact_wide_resnet50
   resnet101
   resnet152
   resnet18
   resnet20_cifar
   resnext101_32x4d
   resnext152_32x4d
   wide_resnet101
   wide_resnet50

Pix2Pix
~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   Pix2PixGenerator
   pix2pix_256
   pix2pix_res_dev
   pix2pix_dev
   PatchDiscriminator
   patch16
   patch34
   patch70
   patch286
   residual_patch34
   residual_patch70
   residual_patch142
   residual_patch286

Pix2PixHD
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   Pix2PixHDGlobalGenerator

StyleGAN2
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   StyleGAN2Generator
   StyleGAN2Discriminator

Other GANs
~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   AutoGAN
   autogan_32
   autogan_64
   autogan_128
   PerceptualNet
   ResidualDiscriminator
   res_discr_3l
   res_discr_4l
   res_discr_5l
   res_discr_6l
   res_discr_7l
   snres_discr_4l
   snres_discr_5l
   snres_discr_6l
   snres_discr_7l
   snres_projdiscr_4l
   snres_projdiscr_5l
   snres_projdiscr_6l
   snres_projdiscr_7l


Image classifiers
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   Attention56Bone
   attention56
   EfficientNet
   Hourglass

Image Transformer
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   UNet
   Hourglass

Classification heads
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   ClassificationHead
   ProjectionDiscr

PixelCNN
~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   PixelCNN
