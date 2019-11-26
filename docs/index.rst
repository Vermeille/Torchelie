Welcome to Torchélie's documentation!
=====================================

.. toctree::
   :caption: Algorithms and training
   :maxdepth: 1

   recipes
   callbacks
   hyper

.. toctree::
   :caption: Pytorch Utils
   :maxdepth: 1

   nn
   optimizers
   utils
   data_learning
   loss
   models
   distributions
   datasets
   transforms
   metrics

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

* :ref:`modindex`
* :ref:`search`


