import torch.nn as nn
from typing import Union, Tuple, Sequence, List, Iterator
from collections import OrderedDict


def tup(x):
    if isinstance(x, (tuple, list)):
        return list(x)
    return [x]


ArgNames = Union[str, List[str]]
NamedModule = Union[nn.Module, Tuple[str, nn.Module]]


class ModuleGraph(nn.Sequential):
    """
    Deprecates CondSeq as ModuleGraph is strictly superior.

    Allows description of networks as computation graphs by labelling inputs
    and outputs of each node. Each node will be ran in declaration order,
    fetching its input values from a pool of named values populated from
    previous node's output values and keyword arguments in forward.

    Simple example:

    >>> m = tnn.ModuleGraph([('x', nn.Linear(10, 20), 'y')], outputs=['y'])
    >>> m(x=torch.randn(1, 10))['y']

    Multiple inputs example:

    If a layer takes more than 1 input, labels can be a tuple or a list of
    labels instead. The same applies if a module returns more than 1 output
    values.

    >>> m = tnn.ModuleGraph([
        ('x0', nn.Linear(10, 20), 'x1'),
        (('x1', 'z'), tnn.AdaIN2d(20, 3), 'y')
        ], outputs=['y'])
    >>> m(x0=torch.randn(1, 10), z=torch.randn(1, 3))['y']

    Named modules:

    Modules can be named as well with a tuple (str, Module).

    >>> m = tnn.ModuleGraph([('x', ('fc', nn.Linear(10, 20)), 'y')], outputs=['y'])
    >>> m.fc
    Linear(10, 20, bias=True)
    """
    def __init__(self, graph: Sequence[Tuple[ArgNames, NamedModule, ArgNames]],
                 outputs: ArgNames) -> None:
        if isinstance(graph[0][1], (tuple, list)):
            super().__init__(OrderedDict([g[1] for g in graph]))
        else:
            super().__init__(*[g[1] for g in graph])
        self.ins = [tup(g[0]) for g in graph]
        self.outs = [tup(g[2]) for g in graph]

        self.outputs = tup(outputs)

    def __iter__(self) -> Iterator[Tuple[ArgNames, nn.Module, ArgNames]]:
        return (x for x in zip(self.ins, self._modules.values(), self.outs))

    def forward(self, **args):
        variables = dict(args)

        for i_names, f, o_names in zip(self.ins, self._modules.values(),
                                       self.outs):
            ins = [variables[k] for k in i_names]
            outs = tup(f(*ins))
            for o, k in zip(outs, o_names):
                variables[k] = o

        return {k: variables[k] for k in self.outputs}

    def to_dot(self) -> None:
        txt = ''
        for i_names, f_nm, o_names in zip(self.ins, self._modules.keys(),
                                       self.outs):
            for k in i_names:
                txt += f'{k} -> {f_nm};\n'
            for k in o_names:
                txt += f'{f_nm} -> {k};\n'
            txt += f'{f_nm} [shape=square];\n'
        return txt
