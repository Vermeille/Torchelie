Understanding recipes
=====================

Torchélie's recipes are what other libraries would call trainers or training
loops. However, Torchélie focuses a lot on making recipes as customizable and
flexible as possible through a declarative interface. But enough nonsense
bullshit, let's jump to the code.

TL;DR: I know time is precious bro. If you just want to see what a real sample
would look like, you can jump to the last code sample on this page, and read
backwards for the explanations you want.

It all starts with a simple Pytorch loop
----------------------------------------

The most minimal pytorch high level training example is as simple as this:

::

   for epoch in range(N_EPOCHS):
      for batch in train_loader:
         process_batch(batch)

But this is not super interesting. The fact is that many times, this loop may
contain the same boilerplate code, and look more like that:

::

   # Initialize some stuff, like the model,  the optimizer, logger etc
   for epoch in range(N_EPOCHS):
      # Prepare for new epoch: zero metrics etc
      for batch in train_loader:
         # Move batch to GPU, zero the gradients etc
         stuff = process_batch(batch)
         # Step optimizer, log some stuff, save the model etc
      # Compute overall epoch metrics, log them, run tests...

That's already more interesting. The general overview is a double loop with
four places (initialization is not included) where you may want to do stuff.
And that stuff is quite often just clutter: logging, computing stats etc.
That's in essence what a Recipe is: a very simple training loop with
attachable callbacks that can react to four events.

Actually, the real source code for Torchelie's Recipe class is that one:

::

     for epoch in range(epochs):
         self.callbacks('on_epoch_start')
         for batch in self.loader:
             self.callbacks.update_state({'batch': batch})
             batch = tu.send_to_device(batch,
                                       self.device,
                                       non_blocking=True)
             self.callbacks.update_state({'batch_gpu': batch})

             self.callbacks('on_batch_start')
             out = self.call_fun(batch)
             out = tu.send_to_device(out, 'cpu', non_blocking=True)
             self.callbacks.update_state(out)
             self.callbacks('on_batch_end')
         self.callbacks('on_epoch_end')
     return self.callbacks.state

Okay, but why is this cool?
---------------------------

Naive code
~~~~~~~~~~

I know what you're thinking: those lines are stupid and you can write them
yourself, it takes less than a minute. This is true. But you also know that
they never remain that simple: you add a ton of things to those lines. Several
events, computations, loggers, snapshotting and whatnots. Let's see from the
simplest example how to benefit from Torchélie's engineering.

To use that barebone Recipe in a simple scenario, let's train a model on the
GPU for 5 epochs of CIFAR10, using as few features as possible for now.

::

   # Let's initialize the model, dataloader, and optimizer
   model = torchvision.models.resnet18(num_classes=10).cuda()
   data_loader = torch.utils.data.DataLoader(CIFAR10(), batch_size=32)
   opt = torchelie.optim.RAdamW(model.parameters(), lr=0.01)

   # define a training pass
   def forward_pass(batch):
      x, y = batch

      opt.zero_grad()  # Don't be angry at me just now
      pred = model(x)
      loss = nn.functional.cross_entropy(pred, y)
      loss.backward()
      opt.step()

      # For now, just return an empty dict
      return {}

   # build our recipe
   recipe = torchelie.recipe.Recipe(forward_pass, data_loader)
   recipe.run(5)

While this work, I would understand that you'd be pissed off: you are still
handling moving the model to the GPU yourself, the boilerplate with the
optimizer is still there, and there is no testing, logging or whatev. It's
kinda shitty. But I gotcha fam.

Handling devices
~~~~~~~~~~~~~~~~

First, let's make the recipe aware that the model exists. If the recipe knows
that the model exists, it will properly handle its serialization and move it
from device to device. We make the recipe aware of stuff by registering things
to it.

::

   # notice that we don't move the model to cuda anymore
   model = torchvision.models.resnet18(num_classes=10)

   # same as before ...

   recipe = torchelie.recipe.Recipe(forward_pass, data_loader)
   # register the model
   recipe.register('model', model)
   # 1) recipe.model now exists
   # 2) recipe.state_dict() will include the model's state_dict
   # 3) moving recipe will move model

   # move the training to gpu
   recipe.cuda()
   recipe.run(5)

You can register as many models, tensors, or objects as you want, they just
need a unique name.

Optimizer Callbacks
~~~~~~~~~~~~~~~~~~~

I promised you to remove common clutter, and yet those stupid optimizer lines
are still there. And it really looks like zeroing the gradients could be a
pre-batch event and stepping a post-batch event, so all in all it should be
unnecessary in the forward pass. And it is.

Let's write our very own optimizer callback!

::

   import torchelie.recipes as tcr

   # There's a better one in Torchelie, you'll actually never do that.
   class OptimizerCallback:
      def __init__(self, opt):
         self.opt = opt

      # For now, just pretend the state argument isn't there
      def on_batch_start(self, state):
         self.opt.zero_grads()

      # For now, just pretend the state argument isn't there
      def on_batch_end(self, state):
         self.opt.step()

   model = torchvision.models.resnet18(num_classes=10)
   opt = torchelie.optim.RAdamW(model.parameters(), lr=0.01)

   def forward_pass(batch):
      x, y = batch
      pred = model(x)
      loss = nn.functional.cross_entropy(pred, y)
      loss.backward()
      return {}

   recipe = tcr.Recipe(forward_pass, data_loader)
   # there's something new here!
   recipe.callbacks.add_callbacks([
      OptimizerCallback(opt)
   ])
   recipe.cuda()
   recipe.run(5)

And now, automagically, the optimizer will do its thing, and the forward pass
now looks as clean as it should. In a real scenario however, you wouldn't write
your own OptimizerCallback class. Torchelie got you covered, and it implements
its own; spiced up with gradient accumulation,  gradient clipping, lr and
momentum logging. But now you know how to write callbacks, and I think it
wasn't that painful :)

Note: I know you're maybe thinking that :code:`loss.backward()` should be
included in a callback as well. But I don't think so. Sometimes you want to do
multiple backwards in one pass, for instance.

Callbacks and state
~~~~~~~~~~~~~~~~~~~

Noticed that empty dict that we're returning? and that mysterious :code:`state`
argument in the callback? While we've not payed much attention to it yet, it's
actually the core of communicating with callbacks. This dict will populate a
state held by the recipe, which callbacks can read and write to. So let's use
it to log the loss

::

   import torchelie.callbacks as tcb

   model = torchvision.models.resnet18(num_classes=10)
   opt = torchelie.optim.RAdamW(model.parameters(), lr=0.01)

   def forward_pass(batch):
      x, y = batch
      pred = model(x)
      loss = nn.functional.cross_entropy(pred, y)
      loss.backward()
      return {'loss': loss}

   recipe = torchelie.recipes.Recipe(forward_pass, data_loader)
   # there's something new here!
   recipe.callbacks.add_callbacks([
      tcb.Counter(),
      tcb.Optimizer(opt),
      tcb.EpochMetricAvg('loss', post_each_batch=True)
      tcb.StdoutLogger(log_every=10),
      tcb.VisdomLogger(visdom_env='main', log_every=10)
   ])
   recipe.cuda()
   recipe.run(5)

Now the forward pass sends to the loss to the shared state. Then callbacks gets
executed in order. So: :code:`EpochMetricAvg` will read the loss value, compute
a running average with all the loss values computed in this epoch so far, and
post it to :code:`state['metrics']['loss']` on each batch.
:code:`state['metrics']` is a conventional place where values are considered
ready (for logging for instance). Every ten iterations, metrics will be
displayed on the standard output and on visdom.

Note: Counter just keeps tracks of the iteration number, epoch number, and
iteration in epoch number, and store that in the state. It is often mandatory
as other callbacks may depend on those, especially loggers. Place it first to
avoid issues.

Now, this starts to look like real Torchélie code.

Composing recipes
~~~~~~~~~~~~~~~~~

While this starts to be pretty cool, we're still missing something real
important: evaluation code. Evaluating is commonly done with a testing set, on
which you compute some metrics, logs the results, etc. Does this sound
familiar? Of course, it's a recipe in itself. Let's write it down.

::

   def test_pass(batch):
      x, y = batch
      with torch.no_grad():
         model.eval()
         pred = model(x)
         loss = nn.functional.cross_entropy(pred, y)
         # no backward in testing
         model.train()
      return {'loss': loss}

   # we have a test set now
   test_recipe = torchelie.recipes.Recipe(test_pass, test_loader)
   test_recipe.callbacks.add_callbacks([
      tcb.Counter(),
      tcb.EpochMetricAvg('loss', post_each_batch=False)
      tcb.StdoutLogger(log_every=-1, prefix='Test'),
      tcb.VisdomLogger(visdom_env='main', log_every=-1, prefix='Test')
   ])

We have a new forward pass disabling gradients and setting the model to eval
mode, a new dataloader, we've removed the Optimizer callback (obviously), and
used other loggers instances that have prefixes in order to avoid name clashes
with the training loop. And logging only happens at the end of the training
loop.

We have two recipes and they're not interacting for now, let's make the
training recipe aware of the testing recipe and call it every 100 iterations.

::

   # Now the training recipe is aware of the testing recipe and handle moving
   # from devices to devices and serialization through `state_dict()`
   recipe.register('testing', test_recipe)
   recipe.callbacks.add_callback([
      tcb.CallRecipe(test_recipe, run_every=100, prefix='test')
   ])

The final code looks just like that:

::

   model = torchvision.models.resnet18(num_classes=10)
   opt = torchelie.optim.RAdamW(model.parameters(), lr=0.01)

   def forward_pass(batch):
      x, y = batch
      pred = model(x)
      loss = nn.functional.cross_entropy(pred, y)
      loss.backward()
      return {'loss': loss}

   def test_pass(batch):
      x, y = batch
      with torch.no_grad():
         model.eval()
         pred = model(x)
         loss = nn.functional.cross_entropy(pred, y)
         model.train()
      return {'loss': loss}

   test_recipe = torchelie.recipes.Recipe(test_pass, test_loader)
   test_recipe.callbacks.add_callbacks([
      tcb.Counter(),
      tcb.EpochMetricAvg('loss', post_each_batch=False)
      tcb.StdoutLogger(log_every=-1, prefix='Test'),
      tcb.VisdomLogger(visdom_env='main', log_every=-1, prefix='Test')
   ])

   recipe = torchelie.recipes.Recipe(forward_pass, data_loader)
   recipe.register('model', model)
   recipe.register('test_recipe', test_recipe)
   recipe.callbacks.add_callbacks([
      tcb.Counter(),
      tcb.Optimizer(opt),
      tcb.EpochMetricAvg('loss', post_each_batch=True)
      tcb.StdoutLogger(log_every=10),
      tcb.VisdomLogger(visdom_env='main', log_every=10)
      tcb.CallRecipe(test_recipe, run_every=100, prefix='test')
   ])
   recipe.cuda()
   recipe.run(5)

Using the predefined recipes and callbacks
==========================================

Even if that is more readable, puts intent first, and is more maintainable,
that's still a lot of boilerplate code. Those loggers, the testing loop and
even the logging of the loss are meant to be be part of the vast majority of
experiments.

I know that and provide several recipes already configured with various
callbacks for common scenarios:

TrainAndTest
  Pretty much what we have covered in this tutorial. It trains on a set, tests
  on another. It even handles for you the need to set the model in eval mode
  and to disable the gradients when testing. It also includes a checkpointing
  callback that will save the recipe's state dict regularly.

  Using this recipe our code reduces to:

::

   model = torchvision.models.resnet18(num_classes=10)
   opt = torchelie.optim.RAdamW(model.parameters(), lr=0.01)

   def forward_pass(batch):
      x, y = batch
      pred = model(x)
      loss = nn.functional.cross_entropy(pred, y)
      loss.backward()
      return {'loss': loss}

   def test_pass(batch):
      x, y = batch
      pred = model(x)
      loss = nn.functional.cross_entropy(pred, y)
      return {'loss': loss}

   recipe = torchelie.recipes.TrainAndTest(model, forward_pass, test_pass,
         data_loader, test_loader)

   recipe.callbacks.add_callbacks([
      tcb.Optimizer(opt),
      tcb.EpochMetricAvg('loss', post_each_batch=True)
   ])
   recipe.test_loop.callbacks.add_callbacks([
      tcb.EpochMetricAvg('loss', post_each_batch=False)
   ])

   recipe.cuda()
   recipe.run(5)

TrainAndCall
  Instead of testing on a test set, it gives you the opportunity to call any
  function. This is what you need if you train a generative model that doesn't
  need any input to work, such as a VAE decoder or a PixelCNN.

Classification
  A TrainAndTest recipe extended with several callbacks: loss logging, accuracy
  logging, confusion matrix generation, image gradient for feature
  visualization, a visual report with best, worst and most confused samples.

  Using this recipe our code reduces to less code that do even more:

::

   model = torchvision.models.resnet18(num_classes=10)
   opt = torchelie.optim.RAdamW(model.parameters(), lr=0.01)

   def forward_pass(batch):
      x, y = batch
      pred = model(x)
      loss = nn.functional.cross_entropy(pred, y)
      loss.backward()
      return {'loss': loss, 'pred': pred}

   def test_pass(batch):
      x, y = batch
      pred = model(x)
      loss = nn.functional.cross_entropy(pred, y)
      return {'loss': loss, 'pred': pred}

   recipe = torchelie.recipes.Classification(model, forward_pass, test_pass,
         data_loader, test_loader, trainset.classes)

   recipe.callbacks.add_callbacks([
      tcb.Optimizer(opt),
   ])

   recipe.cuda()
   recipe.run(5)

CrossEntropyClassification
  Classification recipe that already provides a forward train and test pass, a
  RAdamW optimizer and LR scheduling. Just give it your model, data, and
  hyperparameters and you're good to go without writing a single instruction.

  Using this recipe our code reduces to less code that do even more:

::

   model = torchvision.models.resnet18(num_classes=10)

   recipe = torchelie.recipes.CrossEntropyClassification(model, data_loader,
         test_loader, trainset.classes)

   recipe.cuda()
   recipe.run(5)

Please refer to the recipes' respective documentation for further explanations.
