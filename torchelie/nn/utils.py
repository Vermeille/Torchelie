import torch
from torch.nn import Module

from torch.nn.parameter import Parameter
from typing import Any, TypeVar
from torch.nn.utils.weight_norm import WeightNorm


class WeightScale:
    name: str

    def __init__(self, name: str, scale: float) -> None:
        self.name = name
        self.scale = scale

    def compute_weight(self, module: Module) -> Any:
        g = getattr(module, self.name + '_g')
        return g * self.scale

    @staticmethod
    def apply(module, name: str, scale: float) -> 'WeightScale':
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightScale) and hook.name == name:
                raise RuntimeError("Cannot register two weight_scale hooks on "
                                   "the same parameter {}".format(name))
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError("Cannot register weight_norm "
                                   "and weight_scale hooks on "
                                   "the same parameter {}".format(name))

        fn = WeightScale(name, scale)

        weight = getattr(module, name)
        # remove w from parameter list
        del module._parameters[name]

        module.register_parameter(name + '_g', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))


def weight_scale(module: Module,
                 name: str = 'weight',
                 scale: float = 0) -> Module:
    assert scale != 0, "WeightScale needs a scale != 0"
    WeightScale.apply(module, name, scale)
    return module


def remove_weight_scale(module: Module, name: str = 'weight') -> Module:
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightScale) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_scale of '{}' not found in {}".format(
        name, module))
