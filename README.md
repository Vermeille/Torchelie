# Torchélie

Torchélie is my personal set of helpers, layers, visualization tools and
whatnots that I build around PyTorch.

## torchelie.utils

Functions:

* `freeze` and `unfreeze` that changes `requires_grad` for all tensor in a
  module.

## torchelie.nn

Modules:

* `Debug` doesn't modify its input but prints some statistics. Easy to spot
  exploding or vanishing values.
* `Reshape(*shape)` applies `x.view(x.shape[0], *shape)`.
* `VQ` is a VectorQuantization layer, embedding the VQ-VAE loss in its backward
  pass for a great ease of use.
* `Noise` returns `x + a * z` where `a` is a learnable scalar, and `z` is a
  gaussian noise of the same shape of `x`
* `ImageNetInputNorm` for normalizing images like `torchvision.model` wants it.
* `WithSavedActivations(model, types)` saves all activations of `model` for its
  layers of instance `types` and returns a dict of activations in the forward
  pass instead of just the last value. Forward takes a `detach` boolean
  arguments if the activations must be detached or not.

## torchelie.loss

Modules:

* `PerceptualLoss(l)` is a vgg16 based perceptual loss up to layer number `l`.
  Sum of L1 distances between `x`'s and `y`'s activations in vgg. Only `x` is
  backproped.
