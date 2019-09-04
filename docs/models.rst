Models
======

GAN from Pix2Pix
~~~~~~~~~~~~~~~~


.. automodule:: torchelie.models.patchgan
   :members:

Convolutional
~~~~~~~~~~~~~

.. autofunction:: torchelie.models.ResNetBone
.. autofunction:: torchelie.models.VectorCondResNetBone
.. autofunction:: torchelie.models.ClassCondResNetBone

Image classifiers
~~~~~~~~~~~~~~~~~

.. autofunction:: torchelie.models.ResNetDebug
.. autofunction:: torchelie.models.PreactResNetDebug
.. autofunction:: torchelie.models.VectorCondResNetDebug
.. autofunction:: torchelie.models.ClassCondResNetDebug


Classification heads
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchelie.models.Classifier
   :members:
   :undoc-members:

.. autoclass:: torchelie.models.ProjectionDiscr
   :members:
   :undoc-members:

.. autofunction:: torchelie.models.PerceptualNet

PixelCNN
~~~~~~~~

.. autoclass:: torchelie.models.PixelCNN
   :members:
