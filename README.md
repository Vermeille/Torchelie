# Torchélie

Torchélie is my personal set of helpers, layers, visualization tools and
whatnots that I build around PyTorch.

## torchelie.utils

Contains functions on nets.

* `freeze` and `unfreeze` that changes `requires_grad` for all tensor in a
  module.

## torchelie.nn

Contains modules.

* `Debug` doesn't modify its input but prints some statistics. Easy to spot
  exploding or vanishing values.
* `Reshape(*shape)` applies `.view(*shape)`.
