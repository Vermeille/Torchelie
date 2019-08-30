.. Torchélie documentation master file, created by
   sphinx-quickstart on Tue Aug 27 15:58:29 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Torchélie's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. automodule:: torchelie.optim
   :members:
.. automodule:: torchelie.data_learning
   :members:
.. automodule:: torchelie.lr_scheduler
   :members:
.. automodule:: torchelie.utils
   :members:
.. automodule:: torchelie.datasets
   :members:
.. automodule:: torchelie.loss
   :members:
