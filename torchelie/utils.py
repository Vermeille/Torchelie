import os
import math
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Iterable, TypeVar, Any, overload
from typing import List
from functools import wraps
from inspect import isfunction
from textwrap import dedent


T = TypeVar('T')
T_Module = TypeVar('T_Module', bound=nn.Module)
Numeric = TypeVar('Numeric', torch.Tensor, float)


def fast_zero_grad(net: nn.Module) -> None:
    """
    Set :code:`.grad` to None for all parameters instead of zeroing out. It is
    faster.
    """
    for p in net.parameters():
        p.grad = None


def freeze(net: T_Module) -> T_Module:
    """
    Freeze all parameters of `net`
    """
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def unfreeze(net: T_Module) -> T_Module:
    """
    Unfreeze all parameters of `net`
    """
    for p in net.parameters():
        p.requires_grad_(True)
    return net


def entropy(out: torch.Tensor,
            dim: int = 1,
            reduce: str = 'mean') -> torch.Tensor:
    """
    Compute the entropy of the categorical distribution specified by the logits
    `out` along dimension `dim`.

    Args:
        out (tensor): logits categorial distribution
        dim (int): the dimension along which the distributions are specified
        reduce (str): `"mean"`, `"none"` or `"sum"`
    """
    log_prob = F.log_softmax(out, dim=dim)
    h = -torch.sum(log_prob.exp() * log_prob, dim=dim)
    if reduce == 'none':
        return h
    if reduce == 'mean':
        return h.mean()
    if reduce == 'sum':
        return h.sum()
    assert False, reduce + ' is not a valid reduction method'


def kaiming_gain(m: T_Module,
                 a: float = 0,
                 nonlinearity='leaky_relu',
                 mode='fan_in') -> float:
    """
    Return the std needed to initialize a weight matrix with given parameters.
    """
    fan = nn.init._calculate_correct_fan(m.weight, mode)
    gain = nn.init.calculate_gain(nonlinearity, param=a)
    return gain / math.sqrt(fan)


def kaiming(m: T_Module,
            a: float = 0,
            nonlinearity: str = 'leaky_relu',
            mode: str = 'fan_in',
            dynamic: bool = False) -> T_Module:
    """
    Initialize a module with kaiming normal init

    Args:
        m (nn.Module): the module to init
        a (float): the slope of the nonlinearity
        nonlinearity (str): type of the nonlinearity
        dynamic (bool): wether to scale the weights on the forward pass for
            equalized LR such as ProGAN (default: False)

    Returns:
        the initialized module
    """
    assert isinstance(m.weight, torch.Tensor)
    if nonlinearity in ['relu', 'leaky_relu']:
        if a == 0:
            nonlinearity = 'relu'
        else:
            nonlinearity = 'leaky_relu'

    if not dynamic:
        nn.init.kaiming_normal_(m.weight,
                                a=a,
                                nonlinearity=nonlinearity,
                                mode=mode)
    else:
        from .nn.utils import weight_scale
        nn.init.normal_(m.weight, 0, 1)
        weight_scale(m,
                     scale=kaiming_gain(m,
                                        a=a,
                                        nonlinearity=nonlinearity,
                                        mode=mode))

    if hasattr(m, 'biais') and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


def xavier(m: T_Module,
           a: float = 0,
           nonlinearity: str = 'relu',
           mode: str = 'fan_in',
           dynamic: bool = False) -> T_Module:
    """
    Initialize a module with xavier normal init

    Args:
        m (nn.Module): the module to init
        dynamic (bool): wether to scale the weights on the forward pass for
            equalized LR such as ProGAN (default: False)

    Returns:
        the initialized module
    """
    assert isinstance(m.weight, torch.Tensor)
    if nonlinearity in ['relu', 'leaky_relu']:
        if a == 0:
            nonlinearity = 'relu'
        else:
            nonlinearity = 'leaky_relu'

    if not dynamic:
        nn.init.xavier_normal_(m.weight)
    else:
        from .nn.utils import weight_scale
        nn.init.normal_(m.weight, 0, 1)
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        gain = nn.init.calculate_gain(nonlinearity, param=a)
        weight_scale(m, scale=gain * math.sqrt(2. / (fan_in + fan_out)))

    if hasattr(m, 'biais') and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


def normal_init(m: nn.Module, std: float = 0.02) -> nn.Module:
    """
    Initialize a module with gaussian weights of standard deviation std

    Args:
        m (nn.Module): the module to init

    Returns:
        the initialized module
    """
    assert isinstance(m.weight, torch.Tensor)
    nn.init.normal_(m.weight, 0, std)
    if hasattr(m, 'biais') and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


def constant_init(m: nn.Module, val: float) -> nn.Module:
    """
    Initialize a module with gaussian weights of standard deviation std

    Args:
        m (nn.Module): the module to init

    Returns:
        the initialized module
    """
    assert isinstance(m.weight, torch.Tensor)
    nn.init.constant_(m.weight, val)
    if hasattr(m, 'biais') and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


def nb_parameters(net: nn.Module) -> int:
    """
    Counts the number of parameters of `net`

    Args:
        net (nn.Module): the net

    Returns:
        the number of params
    """
    return sum(p.numel() for p in net.parameters())


def layer_by_name(net: nn.Module, name: str) -> Optional[nn.Module]:
    """
    Get a submodule at any depth of a net by its name

    Args:
        net (nn.Module): the base module containing other modules
        name (str): a name of a submodule of `net`, like `"layer3.0.conv1"`.

    Returns:
        The found layer or `None`
    """
    for layer in net.named_modules():
        if layer[0] == name:
            return layer[1]
    return None


def forever(iterable: Iterable[T]) -> Iterable[T]:
    """
    Cycle through `iterable` forever

    Args:
        iterable (iterable): the iterable
    """
    it = iter(iterable)
    while True:
        try:
            yield next(it)
        except Exception as e:
            print(e)
            it = iter(iterable)


def gram(m: torch.Tensor) -> torch.Tensor:
    """
    Return the Gram matrix of `m`

    Args:
        m (tensor): a matrix of dim 2

    Returns:
        The Gram matrix
    """
    g = torch.einsum('ik,jk->ij', m, m) / m.shape[1]
    return g


def bgram(m: torch.Tensor) -> torch.Tensor:
    """
    Return the batched Gram matrix of `m`

    Args:
        m (tensor): a matrix of dim 3, first one is the batch

    Returns:
        The batch of Gram matrix
    """
    m = m.view(m.shape[0], m.shape[1], -1)
    g = torch.einsum('bik,bjk->bij', m, m) / (m.shape[1] * m.shape[2])
    return g


def dict_by_key(d: Any, k: str) -> Any:
    """
    Recursively index a `dict` by a hierarchical key

    ```
    >>> dict_by_key({'a': [{'b': 42}]}, 'a.0.b')
    42
    ```

    Args:
        d (dict, list, and any level of nesting): the data to index
        k (str): the key

    Returns:
        The value in `d` indexed by `k`
    """
    ks = k.split('.')
    while len(ks) != 0:
        if isinstance(d, dict):
            d = d[ks[0]]
        else:
            d = d[int(ks[0])]
        ks = ks[1:]
    return d


def send_to_device(x: Any, device, non_blocking: bool = False) -> Any:
    """
    Send all tensors contained in `x` to `device`, when `x` is an arbitrary
    nested datastructure of dicts and lists containing tensors

    Args:
        x: the tensors
        device: a torch device
        non_blocking (bool): non blocking

    Returns:
        `x` with device changed
    """
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking)
    elif isinstance(x, list):
        return [
            send_to_device(xx, device, non_blocking=non_blocking) for xx in x
        ]
    elif isinstance(x, tuple):
        return tuple(
            send_to_device(xx, device, non_blocking=non_blocking) for xx in x)
    elif isinstance(x, dict):
        return {
            k: send_to_device(v, device, non_blocking=non_blocking)
            for k, v in x.items()
        }
    return x


def recursive_state_dict(x: Any) -> Any:
    """
    Recursively call state_dict() on all elements contained in a list / tuple /
    dict so that it can be saved safely via torch.save().

    Args:
        x: any nesting of list / tuple / dict containing state_dict()able
            objects

    Returns:
        the same structure with state dicts
    """
    if hasattr(x, 'state_dict'):
        return x.state_dict()
    if isinstance(x, tuple):
        return tuple(recursive_state_dict(xx) for xx in x)
    if isinstance(x, list):
        return [recursive_state_dict(xx) for xx in x]
    if isinstance(x, dict):
        return {k: recursive_state_dict(v) for k, v in x.items()}
    return x


def load_recursive_state_dict(x: Any, obj: Any) -> None:
    """
    Reload a state dict saved with `recursive_state_dict()`

    Args:
        x: the recursive state dict
        obj: the object that has been recursive_state_dict()ed
    """
    if hasattr(obj, 'load_state_dict'):
        obj.load_state_dict(x)
    if isinstance(x, (tuple, list)):
        for xx, oo in zip(x, obj):
            load_recursive_state_dict(xx, oo)
    if isinstance(x, dict):
        for k in obj.keys():
            load_recursive_state_dict(xx[k], oo[k])


class FrozenModule(nn.Module):
    """
    Wrap a module to eval model, can't be turned back to training mode

    Args:
        m (nn.Module): a module
    """
    def __init__(self, m: nn.Module) -> None:
        super(FrozenModule, self).__init__()
        self.m = freeze(m).eval()

    def train(self, mode: bool = True) -> 'FrozenModule':
        return self

    def __getattr__(self, name):
        print(self, name)
        return getattr(super(FrozenModule, self).__getattr__('m'), name)


class DetachedModule:
    """
    Wrap a module to eval model, can't be turned back to training mode, and
    make it invisible to recursive calls on `nn.Module`s

    Args:
        m (nn.Module): a module
    """
    def __init__(self, m):
        self.m = freeze(m).eval()

    def __call__(self, *args, **kwargs):
        return self.m(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.m, name)


@overload
def lerp(a: float, b: float, t: float) -> float:
    ...


@overload
def lerp(a: float, b: float, t: torch.Tensor) -> torch.Tensor:
    ...


@overload
def lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    ...


@overload
def lerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    ...


def lerp(a, b, t):
    r"""
    Linearly interpolate between `a` and `b` according to `t`.


    :math:`(1 - t)a + tb`

    Args:
        a (number or tensor): a
        b (number or tensor): b
        t (number or tensor): t between 0 and 1

    Returns:
        result between a and b
    """
    return (1 - t) * a + t * b


def ilerp(a: Numeric, b: Numeric, t: Numeric) -> Numeric:
    r"""
    Inverse or lerp. For `t` between `a` and `b`, returns the fraction or `a`
    and `b` in `t`.

    :math:`\frac{t - a}{b - a}`

    Args:
        a (number or tensor): a
        b (number or tensor): b
        t (number or tensor): t between a and b

    Returns:
        result
    """
    return (t - a) / (b - a)


def slerp(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
    r"""
    Spherical linear interpolate between `z1` and `z2` according to `t`.

    Args:
        z1 (torch.Tensor): ND tensor, interpolating on last dim
        z2 (torch.Tensor): ND tensor, interpolating on last dim
        t (float): t between 0 and 1

    Returns:
        result between a and b
    """
    z1_l = z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
    z1_n = z1 / z1_l

    z2_l = z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
    z2_n = z2 / z2_l

    dot = torch.sum(z1_n * z2_n, dim=-1).clamp(-1, 1)
    theta_0 = torch.acos(dot)
    theta = t * theta_0

    z3 = z2_n - dot * z1_n
    z3 = z3 / z3.pow(2).sum(dim=-1, keepdim=True).sqrt()

    azimut = lerp(z1_l, z2_l, t)
    return azimut * (z1_n * torch.cos(theta) + z3 * torch.sin(theta))


def as_multiclass_shape(preds, as_probs=False):
    """
    Manipulate an array of logit predictions so that binary prediction is not a
    special case anymore. Outputs `preds` as (Batch size, Num classes (>=2)).

    if preds has one dim, another one is added. If this is binary
    classification, a second column is added as 1 - preds.

    Args:
        preds (tensor): predictions
        as_probs (bool): whether to return the preds as logits or probs

    Returns:
        the predictions reshaped
    """
    assert preds.ndim <= 2

    if preds.ndim == 1:
        preds = preds.unsqueeze(1)
    if preds.shape[1] == 1:
        preds = torch.cat([-preds, preds], dim=1)
        if as_probs:
            preds = torch.sigmoid(preds)
        return preds
    else:
        if as_probs:
            preds = F.softmax(preds, dim=1)
        return preds


class AutoStateDict:
    """
    Inherit this class for automatic :code:`state_dict()` and
    :code:`load_state_dict()` members based on `__dict__`

    Exclusions can be specified via `except_names`
    """
    def __init__(self, except_names: List[str] = []):
        self._except = except_names

    def state_dict(self):
        return {
            key: (val.state_dict() if hasattr(val, 'state_dict') else val)
            for key, val in self.__dict__.items()
            if key not in self._except and key != '_except'
        }

    def load_state_dict(self, state_dict):
        for nm, v in state_dict.items():
            try:
                if hasattr(self.__dict__[nm], 'load_state_dict'):
                    self.__dict__[nm].load_state_dict(v)
                else:
                    self.__dict__[nm] = v
            except KeyError as ke:
                print('no key', ke, 'for', self.__class__.__name__)


def dist_setup(rank):
    """
    initialize a NCCL process group with default port / address. For internal
    use.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl",
                            rank=rank,
                            world_size=torch.cuda.device_count())


class _WrapFun:
    def __init__(self, fun, *args, **kwargs):
        self.fun = fun
        self.args = args
        self.kwargs = kwargs

    def __call__(self, rank):
        dist_setup(rank)
        torch.cuda.set_device(rank)
        torch.backends.cudnn.benchmark = True
        return self.fun(*self.args, **self.kwargs, rank=rank)


def parallel_run(fun, *args, n_gpus: int = torch.cuda.device_count(),
                 **kwargs) -> None:
    """
    Starts a function  in parallel for GPU dispatching.

    Args:
        fun (callable): the function to start, with signature
            :code:`fun(rank, world_size, *args, **kwargs)`. :code:`rank` is the
            GPU id tu use on this thread, and :code:`world_size` is the total
            number of GPUs.
        *args: arguments passed to :code:`fun`
        n_gpus (int, optional): number of GPUs to use. Will default to the
            number of available GPUs.
        **kwargs: kw arguments passed to :code:`fun`
    """
    import torch.multiprocessing as mp
    mp.spawn(_WrapFun(fun, *args, **kwargs, world_size=n_gpus),
             nprocs=n_gpus,
             join=True)


def indent(text: str, amount: int = 4) -> str:
    """
    Indent :code:`text` by :code:`amount` spaces.

    Args:
        text (str): some text
        amount (int): an indentation amount

    Returns:
        indented text
    """
    return '\n'.join((' ' * amount + line) for line in text.splitlines())


def experimental(func):
    """
    Decorator that warns about a function being experimental
    """
    msg = (f'{func.__qualname__}() is experimental, '
           'and may change or be deleted soon if not already broken')

    def deprecate_doc(doc):
        if doc is None:
            return f'**Experimental**\n\n.. warning::\n  {msg}\n\n.\n'
        else:
            return ('**Experimental**: ' + dedent(func.__doc__)
                    + f'.. warning::\n {msg}\n\n\n')

    if isfunction(func):
        func.__doc__ = deprecate_doc(func.__doc__)

        @wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(msg, FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapped
    else:
        cls = func

        def __getstate__(self):
            return super(cls, self).__getstate__()

        def __setstate__(self, state):
            return super(cls, self).__setstate__(state)

        d = {
            '__doc__': deprecate_doc(cls.__doc__),
            '__init__': cls.__init__,
            '__module__': cls.__module__,
            # '__getstate__': __getstate__,
            # '__setstate__': __setstate__
        }
        return type(cls.__name__, (cls, ), d)
