import functools

import torch.nn as nn
from torchelie.utils import layer_by_name


class WithSavedActivations(nn.Module):
    """
    Hook :code:`model` in order to get intermediate activations. The
    activations to save can be either specified by module type or layer name.
    """
    def __init__(self, model, types=(nn.Conv2d, nn.Linear), names=None):
        super(WithSavedActivations, self).__init__()
        self.model = model
        self.activations = {}
        self.detach = True
        self.handles = []

        self.set_keep_layers(types, names)

    def set_keep_layers(self, types=(nn.Conv2d, nn.Linear), names=None):
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
        if self.detach:
            self.activations[name] = output.detach().clone()
        else:
            self.activations[name] = output.clone()

    def forward(self, input, detach: bool):
        """
        Call :code:`self.model(input)`.

        Args:
            input: input to the model
            detach (bool): if True, intermediate activations will be
                :code:`.detach()`d.

        Returns
            model output, a name => activation dict with saved intermediate
            activations.
        """
        self.detach = detach
        self.activations = {}
        out = self.model(input)
        acts = self.activations
        self.activations = {}
        return out, acts
