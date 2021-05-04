import torch
import torch.nn as nn
from torchelie.utils import kaiming_gain, kaiming, fast_zero_grad
from torch.nn import Module
from collections import OrderedDict
from torch.nn.parameter import Parameter
from typing import Any, TypeVar, Callable, Tuple


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


def weight_lambda(
    module: Module,
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
    return weight_lambda(module, 'scale', lambda w: w * scale, name)


def remove_weight_scale(module: Module, name: str = 'weight') -> Module:
    """
    Remove a weight_scale hook previously applied on
    :code:`getattr(module, name)`.
    """
    return remove_weight_lambda(module, 'scale', name)


@torch.no_grad()
def remove_batchnorm(m: nn.Sequential) -> None:
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


@torch.no_grad()
def edit_model(m: T_Module, f: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Allow to edit any part of a model by recursively editing its modules.

    For instance, in order to delete all dropout layers and change relus into
    leakyrelus

    ::

        def make_leaky_no_dropout(m):
            if isinstance(m, nn.ReLU):
                return nn.LeakyReLU(inplace=True)
            if isinstance(m, nn.Dropout2d):
                return nn.Identity()
            return m
        model = edit_model(model, make_leaky_no_dropout)

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


@torch.no_grad()
def weight_norm_and_equal_lr(m: T_Module,
                             leak: float = 0.,
                             mode: str = 'fan_in',
                             init_gain: float = 1.,
                             lr_gain: float = 1.,
                             name: str = 'weight') -> T_Module:
    """
    Set weight norm and equalized learning rate like demodulated convs in
    StyleGAN2 for module m.

    The weight matrix is initialized for a leaky relu nonlinearity of slope a.
    An extra gain can be specified, as well as a differential learning rate
    multiplier.

    See StyleGAN2 paper for more info.

    """
    kai_gain = kaiming_gain(m, a=leak, mode=mode)
    gain = kai_gain * init_gain * lr_gain

    def do_it(w):
        w = w * gain
        shape = w.shape
        w_flat = w.view(w.shape[0], -1)
        w_flat = w_flat / w_flat.norm(dim=1, keepdim=True)
        return w_flat.view(*shape)

    m.weight.data.normal_(0., 1. / lr_gain)
    weight_lambda(m, 'norm_equal_lr', do_it, name=name)
    return m


@torch.no_grad()
def remove_weight_norm_and_equal_lr(module: Module,
                                    name: str = 'weight') -> Module:
    """
    Remove a weight_norm_and_equal_lr hook previously applied on
    :code:`getattr(module, name)`.
    """
    return remove_weight_lambda(module, 'norm_equal_lr', name)


@torch.no_grad()
def insert_after(base: nn.Sequential, key: str, new: nn.Module,
                 name: str) -> nn.Sequential:
    """
    Insert module :code:`new` with name :code:`name` after element :code:`key`
    in sequential :code:`base` and return the new sequence.
    """
    modules_list = list(base._modules.items())
    found = -1
    for i, (nm, m) in enumerate(modules_list):
        if nm == key:
            found = i
            break
    assert found != -1
    modules_list.insert(found + 1, (name, new))
    base._modules = OrderedDict(modules_list)
    return base


@torch.no_grad()
def insert_before(base: nn.Sequential, key: str, new: nn.Module,
                  name: str) -> nn.Sequential:
    """
    Insert module :code:`new` with name :code:`name` before element :code:`key`
    in sequential :code:`base` and return the new sequence.
    """
    modules_list = list(base._modules.items())
    found = -1
    for i, (nm, m) in enumerate(modules_list):
        if nm == key:
            found = i
            break
    assert found != -1
    modules_list.insert(found, (name, new))
    base._modules = OrderedDict(modules_list)
    return base


@torch.no_grad()
def make_leaky(net: nn.Module) -> nn.Module:
    """
    Change all relus into leaky relus for modules and submodules of net.
    """

    def do_it(m: nn.Module) -> nn.Module:
        if isinstance(m, nn.ReLU):
            return nn.LeakyReLU(0.2, m.inplace)
        return m

    return edit_model(net, do_it)


@torch.no_grad()
def net_to_equal_lr(net: nn.Module,
                    leak: float = 0.,
                    mode: str = 'fan_in') -> T_Module:
    """
    Set all Conv2d, ConvTransposed2d and Linear of :code:`net` to equalized
    learning rate, initialized with :func:`torchelie.utils.kaiming` and
    :code:`dynamic=True`.

    Returns:
        :code:`net`.
    """
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            kaiming(m, a=leak, mode=mode, dynamic=True)
    return net


@torch.no_grad()
def net_to_weight_norm_and_equal_lr(net: T_Module,
                                    leak: float = 0.,
                                    mode: str = 'fan_in',
                                    init_gain: float = 1.,
                                    lr_gain: float = 1.) -> T_Module:
    """
    Set all Conv2d, ConvTransposed2d and Linear of :code:`net` to equalized
    learning rate and weight normalization, initialized with
    :func:`torchelie.utils.weight_norm_and_equal_lr`.

    Returns:
        :code:`net`.
    """
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            weight_norm_and_equal_lr(m,
                                     leak=leak,
                                     mode=mode,
                                     init_gain=init_gain,
                                     lr_gain=lr_gain)
    return net


def receptive_field_for(net: nn.Module,
                        input_size: Tuple[int, int, int]) -> Tuple[int, int]:
    """
    Compute the receptive field of :code:`net` using a backward pass.

    .. warning::
        Make sure the net does not have any normalization operation: mean /
        std operations impact the whole volume and make the calculation
        invalid. (BatchNorm is okay in eval mode)

    Args:
        input_size (Tuple[int, int, int]): a (C, H, W) volume size that the
            model can accept. Note that the receptive field calculation is
            bounded by this size.

    Returns:
        height and width receptive fields.
    """
    device = next(iter(net.parameters())).device
    input = torch.randn(1, *input_size, requires_grad=True, device=device)
    net.eval()
    out = net(input)
    net.train()
    if out.dim() == 2:
        out.mean().backward()
    elif out.dim() == 4:
        out[:, :, out.shape[2] // 2, out.shape[3] // 2].mean().backward()
    fast_zero_grad(net)
    has_grad = torch.nonzero(input.grad[0, 0])
    recep_field = has_grad.max(dim=0).values - has_grad.min(dim=0).values + 1
    return recep_field[0].item(), recep_field[1].item()
