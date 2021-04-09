import torch
import torch.nn as nn
from torch.nn import Module

from torch.nn.parameter import Parameter
from typing import Any, TypeVar, Callable
from torch.nn.utils.weight_norm import WeightNorm


class WeightLambda:
    """
    Apply a lambda function as a hook to the weight matrix of a layer before
    a forward pass.

    Don't use it directly, use the functions :code:`weight_lambda()` and
    :code:`remove_weight_lambda()` instead.

    Args:
        hook_name (str): an identifier for that WeightLambda hook, such as
            'l2normalize', 'weight_norm', etc.
        name (str): the name of the module's parameter to apply the hook on
        function (Callable): a function of the form
            :code:`(torch.Tensor) -> torch.Tensor` that takes applies the
            desired computation to the module's parameter.
    """
    name: str

    def __init__(self, hook_name: str, name: str, function) -> None:
        self.name = name
        self.hook_name = hook_name
        self.fun = function

    @staticmethod
    def apply(module, hook_name: str, name: str, function) -> 'WeightLambda':
        fn = WeightLambda(hook_name, name, function)

        weight = getattr(module, name)
        # remove w from parameter list
        del module._parameters[name]

        module.register_parameter(name + '_g', Parameter(weight.data))
        setattr(module, name, fn.fun(getattr(module, name + '_g')))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: Module) -> None:
        weight = self.fun(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.fun(getattr(module, self.name + '_g')))


def weight_lambda(module: Module,
                  hook_name: str,
                  function,
                  name: str = 'weight',
                  ) -> Module:
    """
    Apply :code:`function()` to :code:`getattr(module, name)` on each forward
    pass.

    Allows to implement things such as weight normalization, or equalized
    learning rate weight scaling.

    Args:
        module (nn.Module): the module to hook on
        hook_name (str): an identifier for that WeightLambda hook, such as
            'l2normalize', 'weight_norm', etc.
        function (Callable): a function of the form
            :code:`(torch.Tensor) -> torch.Tensor` that takes applies the
            desired computation to the module's parameter.
        name (str): the name of the module's parameter to apply the hook on.
            Default: 'weight'.

    Returns:
        the module with the hook
    """
    WeightLambda.apply(module, hook_name, name, function)
    return module


def remove_weight_lambda(module: Module,
                         hook_name: str,
                         name: str = 'weight') -> Module:
    """
    Remove the hook :code:`hook_name` applied on member :code:`name` of
    :code:`module`.

    Args:
        module (nn.Module): the module on which the hook has to be removed
        hook_name (str): an identifier for that WeightLambda hook, such as
            'l2normalize', 'weight_norm', etc.
        name (str): the name of the module's parameter the hook is applied on.
            Default: 'weight'.
    """
    for k, hook in module._forward_pre_hooks.items():
        if (isinstance(hook, WeightLambda) and hook.name == name
                and hook.hook_name == hook_name):
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_lambda of '{}' not found in {}".format(
        hook_name, module))

def weight_scale(module: Module,
                 name: str = 'weight',
                 scale: float = 0) -> Module:
    """
    Multiply :code:`getattr(module, name)` by :code:`scale` on forward pass
    as a hook. Used to implement equalized LR for StyleGAN
    """
    return weight_lambda(module, 'scale',  lambda w: w * scale, name)


def remove_weight_scale(module: Module, name: str = 'weight') -> Module:
    """
    Remove a weight_scale hook previously applied on
    :code:`getattr(module, name)`.
    """
    return remove_weight_lambda(module, 'scale', name)


def remove_bn(m: nn.Sequential) -> None:
    """
    Remove BatchNorm in Sequentials / CondSeqs in a smart way, restoring biases
    in the preceding layer.
    """
    ms = list(m._modules.items())

    # transfer biases from BN to previous conv / Linear / Whatever
    for (name1, mod1), (name2, mod2) in zip(ms[:-1], ms[1:]):
        if isinstance(mod2, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if mod1.bias is not None:
                continue

            if mod2.bias is not None:
                with torch.no_grad():
                    mod1.bias = mod2.bias
            else:
                out_ch = len(mod2.running_mean)
                with torch.no_grad():
                    mod1.bias = nn.Parameter(torch.zeros(out_ch))
    # remove bn
    for name, mod in ms:
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            delattr(m, name)


T_Module = TypeVar('T_Module', bound=nn.Module)
def edit_model(m: T_Module, f: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Allow to edit any part of a model by recursively edit its modules.

    For instance, in order to delete all dropout layers and change relus into
    leakyrelus:

    :code:```
    def make_leaky_no_dropout(m):
        if isinstance(m, nn.ReLU):
            return nn.LeakyReLU(inplace=True)
        if isinstance(m, nn.Dropout2d):
            return nn.Identity()
        return m
    model = edit_model(model, make_leaky_no_dropout)
    ```

    Args:
        m (nn.Module): the model to edit
        f (Callabble: nn.Module -> nn.Module): a mapping function applied to
            all modules and submodules

    Returns:
        The edited model.
    """
    for name, mod in m._modules.items():
        m._modules[name] = edit_model(mod, f)
        m._modules[name] = f(mod)
    return f(m)
