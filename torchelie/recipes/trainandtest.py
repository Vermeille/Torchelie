import copy

import torch
import torch.optim as optim
import torchvision.models as tvmodels

import torchelie.metrics.callbacks as cb
import torchelie.utils as tu

from torchelie.recipes.trainandcallbase import TrainAndCallBase

class TrainAndTest(TrainAndCallBase):
    """
    Vanilla training loop and testing loop. Displays Loss.

    It logs the same things as TrainAndCallBase, plus an averaged loss over the
    testing data, logged to visdom and stdout, and any metric returned by
    `model.validation_step()`.

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
        super(TrainAndTest, self).__init__(model=model,
                                           visdom_env=visdom_env,
                                           test_every=test_every,
                                           train_callbacks=train_callbacks,
                                           device=device)
        self.test_state = {'metrics': {}}
        self.test_callbacks = cb.CallbacksRunner(test_callbacks + [
            cb.EpochMetricAvg('loss', False),
            cb.VisdomLogger(
                visdom_env=visdom_env, log_every=-1, prefix='test_'),
            cb.StdoutLogger(log_every=-1, prefix='Test'),
        ])

    def after_train(self):
        test_loader = self.test_loader
        with torch.no_grad():
            self.test_state = {
                    'iters': self.state['iters'],
                    'epoch': self.state['epoch'],
                    'epoch_batch': self.state['epoch_batch']}
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
        self.test_loader = test_loader
        train_res = super(TrainAndTest, self).__call__(train_loader, epochs)
        return self.model, self.state['test_metrics']
