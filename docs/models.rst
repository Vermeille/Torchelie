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

Other GANs
~~~~~~~~~~

.. autoclass:: torchelie.models.AutoGAN
   :members:
.. autofunction:: torchelie.models.autogan_32
.. autofunction:: torchelie.models.autogan_64
.. autofunction:: torchelie.models.autogan_128

.. autofunction:: torchelie.models.snres_discr
.. autofunction:: torchelie.models.snres_projdiscr
.. autofunction:: torchelie.models.snres_discr_4l
.. autofunction:: torchelie.models.snres_projdiscr_4l
.. autofunction:: torchelie.models.snres_discr_5l
.. autofunction:: torchelie.models.snres_projdiscr_5l

Convolutional
~~~~~~~~~~~~~

.. autofunction:: torchelie.models.VggBNBone
.. autofunction:: torchelie.models.ResNetBone
.. autofunction:: torchelie.models.VectorCondResNetBone
.. autofunction:: torchelie.models.ClassCondResNetBone
.. autoclass:: torchelie.models.UNetBone
.. autoclass:: torchelie.models.Attention56Bone

Image classifiers
~~~~~~~~~~~~~~~~~

.. autofunction:: torchelie.models.VggDebug
.. autofunction:: torchelie.models.ResNetDebug
.. autofunction:: torchelie.models.PreactResNetDebug
.. autofunction:: torchelie.models.VectorCondResNetDebug
.. autofunction:: torchelie.models.ClassCondResNetDebug
.. autofunction:: torchelie.models.attention56

Image Segmenter
~~~~~~~~~~~~~~~

.. autofunction:: torchelie.models.UNet
.. autoclass:: torchelie.models.Hourglass

Classification heads
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchelie.models.Classifier
   :members:
   :undoc-members:

.. autoclass:: torchelie.models.Classifier1
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
