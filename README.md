# Torchélie

Torchélie is my personal set of helpers, layers, visualization tools and
whatnots that I build around PyTorch.

## `torchelie.utils`

Functions:

* `freeze` and `unfreeze` that changes `requires_grad` for all tensor in a
  module.
* `entropy(x, dim, reduce)` computes the entropy of `x` along dimension `dim`,
  assuming it represents the unnormalized probabilities of a categorial
  distribution.

## `torchelie.nn`

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
* `MaskedConv2d` is a masked convolution for PixelCNN
* `BatchNorm2d`, `NoAffineBatchNorm2d` should be strictly equivalent to
  Pytorch's, and `ConditionalBN2d` gets its weight and bias parameter from a
  linear projection of a `z` vector.
* `MovingAverageBN2d`, `NoAffineMABN2d` and `ConditionalMABN2d` are the same as
  above, except they also use moving average of the statistics at train time
  for greater stability. Useful ie for GANs if you can't use a big ass batch
  size and BN introduces too much noise.
* `Conv2d`, `Conv3x3`, `Conv1x1`, `Conv2dBNReLU`, `Conv2dCondBNReLU`, etc. Many
  different convenience blocks in `torchelie.nn.blocks.py`

## `torchelie.loss`

Modules:

* `PerceptualLoss(l)` is a vgg16 based perceptual loss up to layer number `l`.
  Sum of L1 distances between `x`'s and `y`'s activations in vgg. Only `x` is
  backproped.

Functions:

* `ortho(x)` applies an orthogonal regularizer as in _Brock et al (2018)_
  (BigGAN)

### `torchelie.loss.gan`

Each submodule is a GAN loss function. They all contain three methods:
`real(x)` and `fake(x)` to train the discriminator, and `ŋenerated(x)` to
improve the Generator.

Available:

* Standard loss (BCE)
* Hinge

## `torchelie.transforms`

Torchvision-like transforms:

* `ResizeNoCrop` resizes the _longest_ border of an image ot a given size,
  instead of torchvision that resize the smallest side. The image is then
  _smaller_ than the given size and needs padding for batching.
* `AdaptPad` pads an image so that it fits the target size.


## `torchelie.lr_scheduler`

Classes:

* `CurriculumScheduler` takes a lr schedule and an optimizer as argument. Call
  `sched.step()` on each batch. The lr will be interpolated linearly between
  keypoints.
