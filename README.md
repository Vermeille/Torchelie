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
* `Reshape(*shape)` applies `x.view(x.shape[0], *shape)`.
* `VQ` is a VectorQuantization layer, embedding the VQ-VAE loss in its backward
  pass for a great ease of use.
* `Noise` returns `x + a * z` where `a` is a learnable scalar, and `z` is a
  gaussian noise of the same shape of `x`
