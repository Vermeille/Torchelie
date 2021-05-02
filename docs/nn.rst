torchelie.nn
============

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

   ConvBlock
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

torchelie.nn.utils
==================

.. currentmodule:: torchelie.nn.utils

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   receptive_field_for

Model edition
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   edit_model
   insert_after
   insert_before
   make_leaky
   remove_batchnorm

Lambda
~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   WeightLambda
   weight_lambda
   remove_weight_lambda

Weight normalization / equalized learning rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: klass.rst
   :nosignatures:

   weight_norm_and_equal_lr
   remove_weight_norm_and_equal_lr
   remove_weight_scale
   weight_scale
   net_to_equal_lr
   net_remove_weight_norm_and_equal_lr

