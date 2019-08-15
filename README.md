# Torchélie

<img src="https://github.com/Vermeille/Torchelie/blob/master/logo.png" height="200"/>

Torchélie is my personal set of helpers, layers, visualization tools and
whatnots that I build around PyTorch.

Feel free to use this code or just even copy paste the bits you need. If you
want to submit a PR, you're welcome, but talk to me first as I may not accept
it if I don't think the value I get is not worth the maintenance cost. This is
my own set of helpers that I share, not a PyTorch library that tries to fit
everyone's use.

## `torchelie.recipes`

Classes implementing full algorithms, from training to usage

* `NeuralStyleRecipe` implements Gatys' Neural Artistic Style. Also directly
  usable with commandline with `python3 -m torchelie.recipes.neural_style`
* `FeatureVisRecipe` implements feature visualization through backprop. The
  image is implemented in Fourier space which makes it powerful (see
  [this](https://distill.pub/2018/differentiable-parameterizations/) and
  [that](https://distill.pub/2017/feature-visualization/) ). Usable as
  commandline as well with `python -m torchelie.recipes.feature_vis`.
* `DeepDreamRecipe` implements something close to Deep Dream.
  `python -m torchelie.recipes.deepdream` works.

## `torchelie.utils`

Functions:

* `freeze` and `unfreeze` that changes `requires_grad` for all tensor in a
  module.
* `entropy(x, dim, reduce)` computes the entropy of `x` along dimension `dim`,
  assuming it represents the unnormalized probabilities of a categorial
  distribution.
* `kaiming(m)` / `xavier(m)` returns `m` after a kaiming / xavier
  initialization of `m.weight`
* `nb_parameters` returns the number of trainables parameters in a module
* `layer_by_name` finds a module by its (instance) name in a module
* `gram` / `bgram` compute gram and batched gam matrices.

## `torchelie.nn`

Debug modules:

* `Dummy` does nothing to its input.
* `Debug` doesn't modify its input but prints some statistics. Easy to spot
  exploding or vanishing values.

Normalization modules:

* `ImageNetInputNorm` for normalizing images like `torchvision.model` wants
  them.
* `MovingAverageBN2d`, `NoAffineMABN2d` and `ConditionalMABN2d` are the same as
  above, except they also use moving average of the statistics at train time
  for greater stability. Useful ie for GANs if you can't use a big ass batch
  size and BN introduces too much noise.
* `AdaIN2d` is adaptive instancenorm for style transfer and stylegan.
* `Spade2d` / `MovingAverageSpade2d`, for GauGAN.
* `PixelNorm` from ProGAN and StyleGAN.
* `BatchNorm2d`, `NoAffineBatchNorm2d` should be strictly equivalent to
  Pytorch's, and `ConditionalBN2d` gets its weight and bias parameter from a
  linear projection of a `z` vector.

Misc modules:

* `FiLM2d` is affine conditioning `f(z) * x + g(z)`.
* `Noise` returns `x + a * z` where `a` is a learnable scalar, and `z` is a
  gaussian noise of the same shape of `x`
* `Reshape(*shape)` applies `x.view(x.shape[0], *shape)`.
* `VQ` is a VectorQuantization layer, embedding the VQ-VAE loss in its backward
  pass for a great ease of use.

Container modules:

* `ConditionalSequential` is an extension of `nn.Sequential` that also applies a
  second input on the layers having `condition()`

Model manipulation modules:

* `WithSavedActivations(model, types)` saves all activations of `model` for its
  layers of instance `types` and returns a dict of activations in the forward
  pass instead of just the last value. Forward takes a `detach` boolean
  arguments if the activations must be detached or not.

Net Blocks:

* `MaskedConv2d` is a masked convolution for PixelCNN
* `Conv2d`, `Conv3x3`, `Conv1x1`, `Conv2dBNReLU`, `Conv2dCondBNReLU`, etc. Many
  different convenience blocks in `torchelie.nn.blocks.py`
* `ResNetBlock`, `PreactResNetBlock`
* `ResBlock` is a classical residual block with batchnorm
* `ClassConditionalResBlock`
* `ConditionalResBlock` instead uses `ConditionalBN2d`
* `SpadeResBlock` instead uses `Spade2d`

## `torchelie.models`

* `VggBNBone` is a parameterizable stack of convs vgg style. Look at `VggDebug`
  for its usage.
* `ResNetBone` for resnet style bone.
* `Classifier` adds two linear layers to a bone for classification.
* `Patch16`, `Patch32`, `Patch70`, `Patch286` are Pix2Pix's PatchGAN's
  discriminators
* `PerceptualNet` is a VGG16 with correctly named layers for more convenient
  use with `WithSavedActivations`

Debug models:

* `VggDebug`
* `ResNetDebug`
* `PreactResNetDebug`

## `torchelie.loss`

Modules:

* `PerceptualLoss(l)` is a vgg16 based perceptual loss up to layer number `l`.
  Sum of L1 distances between `x`'s and `y`'s activations in vgg. Only `x` is
  backproped.
* `NeuralStyleLoss`

Functions:

* `ortho(x)` applies an orthogonal regularizer as in _Brock et al (2018)_
  (BigGAN)
* `total_variation(x)` applies a spatial L1 loss on 2D tensors

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
* `Canny` runs canny edge detector (requires OpenCV)
* `MultiBranch` allows different transformations branches in order to transform
  the same image in different ways. Useful for self supervision tasks for
  instance.

## `torchelie.lr_scheduler`

Classes:

* `CurriculumScheduler` takes a lr schedule and an optimizer as argument. Call
  `sched.step()` on each batch. The lr will be interpolated linearly between
  keypoints.

## `torchelie.datasets.debug`

* `ColoredColumns` / `ColoredRows` are datasets of precedurally generated
  images of rows / columns randomly colorized.

## `torchelie.metrics`

* `WindowAvg`: averages measures over a k-long sequence
* `ExponentialAvg`: applies an exponential averaging method over measures
* `RunningAvg`: accumulates total number of items and sum to provide an
  accurate average estimation

## `torchelie.data_learning`

Data parameterization for optimization, like neural style or feature viz.

Modules:

* `PixelImage` an image to be optimized.
* `SpectralImage` an image Fourier-parameterized to ease optimization.
* `CorrelateColors` assumes the input is an image with decorrelated color
  components. It correlates back the color using some ImageNet precomputed
  correlation statistics to ease optimization.

# Testing

* `classification.py` tests bones for classifiers on MNIST or CIFAR10
* `conditional.py` tests class conditional layers with a conditional
  classification task `argmin L(f(x, z), y)` where `x` is a MNIST sample, `z` a
  class label, and `y = 1` if `z` is the correct label for `x`, 0 otherwise.

