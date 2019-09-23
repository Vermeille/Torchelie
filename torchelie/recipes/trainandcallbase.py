import torchelie.metrics.callbacks as cb
import torchelie.utils as tu


class TrainAndCallBase:
    """
    Vanilla training loop and testing loop BASE. Inherit it and fill
    `after_train()`. Fill `self.test_state` there.

    It will:

    - log the loss, with a running average over each epoch
    - log metrics to visdom (if applicable)
    - log metrics to stdout
    - save the model, optimizer and metrics after each epoch

    Args:
        model (nn.Model): a model
            The model must define:
                - `model.make_optimizer()`
                - `model.train_step(batch, opt)` returns a dict with key loss
        visdom_env (str): name of the visdom environment to use, or None for
            not using Visdom (default: None)
        test_every (int): testing frequency, in number of iterations (default:
            1000)
        train_callbacks (list of Callback): additional training callbacks
            (default: [])
        device: a torch device (default: 'cpu')
    """

    def __init__(self,
                 model,
                 visdom_env=None,
                 test_every=1000,
                 train_callbacks=[],
                 device='cpu'):
        self.device = device
        self.model = model.to(device)
        self.opt = model.make_optimizer()
        self.test_every = test_every
        self.state = {'metrics': {}}
        self.test_state = {'metrics': {}}
        self.train_callbacks = cb.CallbacksRunner(train_callbacks + [
            cb.WindowedMetricAvg('loss'),
            cb.VisdomLogger(visdom_env=visdom_env, log_every=100),
            cb.StdoutLogger(log_every=100),
            cb.Checkpoint(
                (visdom_env or 'training') + '/model', {
                    'model': self.model,
                    'opt': self.opt,
                    'metrics': self.state['metrics'],
                    'test_metrics': self.test_state['metrics']
                })
        ])

    def after_train(self):
        raise NotImplementedError(
            "You cannot use TrainAndCallBase, inherit it")

    def __call__(self, train_loader, epochs=5):
        """
        Runs the recipe.

        Args:
            train_loader (DataLoader): Training set dataloader
            epochs (int): number of epochs

        Returns:
            trained model, test metrics
        """
        self.state['iters'] = 0
        for epoch in range(epochs):
            self.state['epoch'] = epoch
            self.train_callbacks('on_epoch_start', self.state)
            for i, batch in enumerate(train_loader):
                self.state['epoch_batch'] = i
                self.state['batch'] = batch
                batch = tu.send_to_device(batch,
                                          self.device,
                                          non_blocking=True)
                self.state['batch_gpu'] = batch

                self.train_callbacks('on_batch_start', self.state)

                out = self.model.train_step(batch, self.opt)
                out = tu.send_to_device(out, 'cpu', non_blocking=True)
                self.state.update(out)

                self.train_callbacks('on_batch_end', self.state)

                if self.state['iters'] % self.test_every == 0:
                    self.model.eval()
                    metrics = self.after_train()
                    self.state['test_metrics'] = metrics
                    self.model.train()
                self.state['iters'] += 1
            self.train_callbacks('on_epoch_end', self.state)

        self.model.eval()
        return self.model, self.state['test_metrics']
