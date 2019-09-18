Models
======

GAN from GauGAN
~~~~~~~~~~~~~~~

.. autoclass:: torchelie.models.VggImg2ImgGeneratorDebug
   :members:
   :undoc-members:

.. autoclass:: torchelie.models.VggClassCondGeneratorDebug
   :members:
   :undoc-members:

GAN from Pix2Pix
~~~~~~~~~~~~~~~~


.. autofunction:: torchelie.models.patch_discr
.. autofunction:: torchelie.models.proj_patch_discr
.. autofunction:: torchelie.models.Patch286
.. autofunction:: torchelie.models.Patch70
.. autofunction:: torchelie.models.Patch32
.. autofunction:: torchelie.models.ProjPatch32
.. autofunction:: torchelie.models.Patch16

FIXME: Where to put that one?
.. autofunction:: torchelie.models.VggGeneratorDebug

Convolutional
~~~~~~~~~~~~~

.. autofunction:: torchelie.models.VggBNBone
.. autofunction:: torchelie.models.ResNetBone
.. autofunction:: torchelie.models.VectorCondResNetBone
.. autofunction:: torchelie.models.ClassCondResNetBone
.. autoclass:: torchelie.models.UNetBone

Image classifiers
~~~~~~~~~~~~~~~~~

.. autofunction:: torchelie.models.VggDebug
.. autofunction:: torchelie.models.ResNetDebug
.. autofunction:: torchelie.models.PreactResNetDebug
.. autofunction:: torchelie.models.VectorCondResNetDebug
.. autofunction:: torchelie.models.ClassCondResNetDebug

Image Segmenter
~~~~~~~~~~~~~~~

.. autofunction:: torchelie.models.UNet

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
