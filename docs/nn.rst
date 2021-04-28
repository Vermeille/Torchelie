Layers
======

Convolutions
~~~~~~~~~~~~

.. currentmodule:: torchelie.nn
.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   Conv2d
   Conv3x3
   Conv1x1
   MaskedConv2d
   TopLeftConv2d

Normalization
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   AdaIN2d
   FiLM2d
   PixelNorm
   ImageNetInputNorm
   ConditionalBN2d
   Spade2d
   AttenNorm2d

Misc
~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   VQ
   MultiVQ
   Noise
   Debug
   Dummy
   Lambda
   Reshape
   Interpolate2d
   InterpolateBilinear2d
   AdaptiveConcatPool2d
   ModulatedConv
   SelfAttention2d
   GaussianPriorFunc
   UnitGaussianPrior
   InformationBottleneck
   Const
   SinePositionEncoding2d
   MinibatchStddev

Blocks
~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   MConvNormReLU
   MConvBNReLU
   SpadeResBlock
   AutoGANGenBlock
   ResidualDiscrBlock
   StyleGAN2Block
   SEBlock
   PreactResBlock
   PreactResBlockBottleneck
   ResBlock
   ResBlockBottleneck
   ConvDeconvBlock
   UBlock


Sequential
~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   WithSavedActivations
   CondSeq
   ModuleGraph

Activations
~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   HardSigmoid
   HardSwish

Utils
=====

Model edition
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   utils.edit_model
   utils.insert_after
   utils.insert_before
   utils.make_leaky
   utils.remove_batchnorm

Lambda
~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   utils.WeightLambda
   utils.weight_lambda
   utils.remove_weight_lambda

Weight normalization / equalized learning rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   utils.weight_norm_and_equal_lr
   utils.remove_weight_norm_and_equal_lr
   utils.remove_weight_scale
   utils.weight_scale

