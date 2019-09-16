import copy

import torch
import torch.optim as optim
import torchvision.models as tvmodels

import torchelie.metrics.callbacks as cb
import torchelie.utils as tu


class TrainAndTest:
    """
    Vanilla training loop and testing loop. Displays Loss.

    Args:
        model (nn.Model): a model
            The model must define:

                - `model.make_optimizer()`
                - `model.validation_step(batch)` returns a dict with key loss
                - `model.train_step(batch, opt)` returns a dict with key loss
        visdom_env (str): name of the visdom environment to use, or None for
            not using Visdom (default: None)
        test_every (int): testing frequency, in number of iterations (default:
            1000)
        train_callbacks (list of Callback): additional training callbacks
            (default: [])
        test_callbacks (list of Callback): additional testing callbacks
            (default: [])
        device: a torch device (default: 'cpu')
    """
    def __init__(self,
                 model,
                 visdom_env=None,
                 test_every=1000,
                 train_callbacks=[],
                 test_callbacks=[],
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
        ])
        self.test_callbacks = cb.CallbacksRunner(test_callbacks + [
            cb.EpochMetricAvg('loss', False),
            cb.VisdomLogger(
                visdom_env=visdom_env, log_every=-1, prefix='test_'),
            cb.StdoutLogger(log_every=-1, prefix='Test'),
            cb.Checkpoint(
                'models/clf', {
                    'model': self.model,
                    'opt': self.opt,
                    'metrics': self.state['metrics'],
                    'test_metrics': self.test_state['metrics']
                })
        ])

    def evaluate(self, test_loader):
        with torch.no_grad():
            self.test_state = self.state
            self.test_state['metrics'] = {}
            self.test_callbacks('on_epoch_start', self.test_state)
            for batch in test_loader:
                self.test_state['batch'] = batch
                batch = tu.send_to_device(batch,
                                          self.device,
                                          non_blocking=True)
                self.test_state['batch_gpu'] = batch

                self.test_callbacks('on_batch_start', self.test_state)
                out = self.model.validation_step(batch)
                out = tu.send_to_device(out, 'cpu', non_blocking=True)
                self.test_state.update(out)
                self.test_callbacks('on_batch_end', self.test_state)

            self.test_callbacks('on_epoch_end', self.test_state)
            return copy.deepcopy(self.test_state['metrics'])

    def __call__(self, train_loader, test_loader, epochs=5):
        """
        Runs the recipe.

        Args:
            train_loader (DataLoader): Training set dataloader
            test_loader (DataLoader): Testing set dataloader
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
                    metrics = self.evaluate(test_loader)
                    self.state['test_metrics'] = metrics
                    self.model.train()
                self.state['iters'] += 1
            self.train_callbacks('on_epoch_end', self.state)

        self.model.eval()
        return self.model, self.state['test_metrics']
