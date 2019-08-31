Welcome to Torchélie's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   optimizers

Torchélie aims to provide new utilities, layers, tools to pytorch users,
as an extension library. It aims for minimal codependence between components,
so that you can use only what you need with a strong pytorch flavour, without
learning a new tool.

It provides:

- layers
- models (untrained)
- new optimizers and schedulers
- utility functions
- datasets and dataset utilities
- losses
- and some more specific things

Torchélie also tries to meet the likes of Ignite and Pytorch-Lightning by
providing training loops with automatic logging, averaging, checkpointing, and
visualisation. Unlike those however, instead of providing a "one size fits all"
training loop, Torchélie aims to make writing them easy.

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Utils
-----

.. automodule:: torchelie.utils
   :members:

Data Learning
-------------

.. automodule:: torchelie.data_learning
   :members:

Datasets
--------

.. autoclass:: torchelie.datasets.HorizontalConcatDataset
   :members:
.. autoclass:: torchelie.datasets.PairedDataset
   :members:
.. autoclass:: torchelie.datasets.MixUpDataset
   :members:
.. autofunction:: torchelie.datasets.mixup
.. autoclass:: torchelie.datasets.NoexceptDataset
   :members:

Loss
----

.. autoclass:: torchelie.loss.OrthoLoss
   :members:
.. autofunction:: torchelie.loss.ortho
.. autoclass:: torchelie.loss.TotalVariationLoss
   :members:
.. autofunction:: torchelie.loss.total_variation
.. autoclass:: torchelie.loss.ContinuousCEWithLogits
   :members:
.. autofunction:: torchelie.loss.continuous_cross_entropy
.. autoclass:: torchelie.loss.FocalLoss
   :members:
.. autofunction:: torchelie.loss.focal_loss
.. autoclass:: torchelie.loss.PerceptualLoss
   :members:
.. autoclass:: torchelie.loss.NeuralStyleLoss
   :members:
