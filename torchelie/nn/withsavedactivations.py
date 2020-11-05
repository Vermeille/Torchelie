import functools

import torch
import torch.nn as nn
from torchelie.utils import layer_by_name, freeze


class WithSavedActivations(nn.Module):
    """
    FIXME: PLZ DOCUMENT ME
    """
    def __init__(self, model, types=(nn.Conv2d, nn.Linear), names=None):
        """
        Initialize model.

        Args:
            self: (todo): write your description
            model: (todo): write your description
            types: (todo): write your description
            nn: (int): write your description
            Conv2d: (todo): write your description
            nn: (int): write your description
            Linear: (todo): write your description
            names: (list): write your description
        """
        super(WithSavedActivations, self).__init__()
        self.model = model
        self.activations = {}
        self.detach = True
        self.handles = []

        self.set_keep_layers(types, names)


    def set_keep_layers(self, types=(nn.Conv2d, nn.Linear), names=None):
        """
        Set keep keep layers of layers.

        Args:
            self: (todo): write your description
            types: (str): write your description
            nn: (todo): write your description
            Conv2d: (todo): write your description
            nn: (todo): write your description
            Linear: (todo): write your description
            names: (str): write your description
        """
        for h in self.handles:
            h.remove()

        if names is None:
            for name, layer in self.model.named_modules():
                if isinstance(layer, types):
                    h = layer.register_forward_hook(functools.partial(
                        self._save, name))
                    self.handles.append(h)
        else:
            for name in names:
                layer = layer_by_name(self.model, name)
                h = layer.register_forward_hook(functools.partial(
                    self._save, name))
                self.handles.append(h)


    def _save(self, name, module, input, output):
        """
        Save the given module.

        Args:
            self: (todo): write your description
            name: (str): write your description
            module: (todo): write your description
            input: (array): write your description
            output: (todo): write your description
        """
        if self.detach:
            self.activations[name] = output.detach().clone()
        else:
            self.activations[name] = output.clone()

    def forward(self, input, detach):
        """
        Evaluate the model.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            detach: (todo): write your description
        """
        self.detach = detach
        self.activations = {}
        out = self.model(input)
        acts = self.activations
        self.activations = {}
        return out, acts
