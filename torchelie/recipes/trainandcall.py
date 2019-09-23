import copy

import torch

import torchelie.metrics.callbacks as cb
import torchelie.utils as tu

from torchelie.recipes.trainandcallbase import TrainAndCallBase


class TrainAndCall(TrainAndCallBase):
    """
    Training loop, calls `model.after_train()` after every `test_every`
    iterations. Displays Loss.

    It logs the same things as TrainAndCallBase, plus whatever is returned by
    `model.after_train()`

    Args:
        model (nn.Model): a model
            The model must define:
                - `model.make_optimizer()`
                - `model.after_train()` returns a dict
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
        super(TrainAndCall, self).__init__(model=model,
                                           visdom_env=visdom_env,
                                           test_every=test_every,
                                           train_callbacks=train_callbacks,
                                           device=device)
        self.test_state = {'metrics': {}}
        self.test_callbacks = cb.CallbacksRunner(test_callbacks + [
            cb.VisdomLogger(
                visdom_env=visdom_env, log_every=-1, prefix='test_'),
            cb.StdoutLogger(log_every=-1, prefix='Test'),
        ])

    def after_train(self):
        with torch.no_grad():
            self.test_state = self.state
            self.test_state['metrics'] = {}
            self.test_callbacks('on_epoch_start', self.test_state)

            out = self.model.after_train()
            out = tu.send_to_device(out, 'cpu', non_blocking=True)
            self.test_state.update(out)

            self.test_callbacks('on_epoch_end', self.test_state)
            return copy.deepcopy(self.test_state['metrics'])

    def __call__(self, train_loader, epochs=5):
        """
        Runs the recipe.

        Args:
            train_loader (DataLoader): Training set dataloader
            epochs (int): number of epochs

        Returns:
            trained model, test metrics
        """
        train_res = super(TrainAndCall, self).__call__(train_loader, epochs)
        return train_res, self.state['test_metrics']

