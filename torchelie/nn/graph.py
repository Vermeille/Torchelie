import torch.nn as nn
from typing import Union, Tuple, List
from torchelie.utils import experimental


def tup(x):
    if isinstance(x, (tuple, list)):
        return list(x)
    return [x]


ArgNames = Union[str, List[str]]
NamedModule = Tuple[str, nn.Module]


class ModuleGraph(nn.Sequential):
    """
    Allows description of networks as computation graphs. The graph is
    constructed by labelling inputs and outputs of each node. Each node will be
    ran in declaration order, fetching its input values from a pool of named
    values populated from previous node's output values and keyword arguments
    in forward.

    Simple example:

    >>> m = tnn.ModuleGraph(outputs='y')
    >>> m.add_operation(
            inputs=['x'],
            operation=nn.Linear(10, 20),
            name='linear',
            outputs=['y'])
    >>> m(x=torch.randn(1, 10))
    <a bunch of numbers>

    Multiple inputs example:

    If a layer takes more than 1 input, labels can be a tuple or a list of
    labels instead. The same applies if a module returns more than 1 output
    values.

    >>> m = tnn.ModuleGraph(outputs=['x1', 'y'])
    >>> m.add_operation(
            inputs=['x0'],
            operation=nn.Linear(10, 20)
            name='linear',
            outputs=['x1'])
    >>> m.add_operation(
            inputs=['x1', 'z'],
            operation=nn.AdaIN2d(20, 3)
            name='adain',
            outputs=['y'])
    >>> m(x0=torch.randn(1, 10), z=torch.randn(1, 3))['y']
    <a bunch of numbers>
    """
    def __init__(self, outputs: Union[str, List[str]]) -> None:
        super().__init__()
        self.ins: List[List[str]] = []
        self.outs: List[List[str]] = []

        self.outputs = outputs

    def add_operation(self, inputs: List[str], outputs: List[str], name: str,
                      operation: nn.Module) -> 'ModuleGraph':
        self.ins.append(inputs)
        self.outs.append(outputs)
        self.add_module(name, operation)
        return self

    def forward(self, **args):
        variables = dict(args)

        for i_names, f, o_names in zip(self.ins, self._modules.values(),
                                       self.outs):
            ins = [variables[k] for k in i_names]
            outs = tup(f(*ins))
            for o, k in zip(outs, o_names):
                variables[k] = o

        if isinstance(self.outputs, str):
            return variables[self.outputs]
        return {k: variables[k] for k in self.outputs}

    @experimental
    def to_dot(self) -> str:
        txt = ''
        for i_names, f_nm, o_names in zip(self.ins, self._modules.keys(),
                                          self.outs):
            for k in i_names:
                txt += f'{k} -> {f_nm};\n'
            for k in o_names:
                txt += f'{f_nm} -> {k};\n'
            txt += f'{f_nm} [shape=square];\n'
        return txt
