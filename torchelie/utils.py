import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze(net):
    """
    Freeze all parameters of `net`
    """
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def unfreeze(net):
    """
    Unfreeze all parameters of `net`
    """
    for p in net.parameters():
        p.requires_grad_(True)
    return net


def entropy(out, dim=1, reduce='mean'):
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


def kaiming(m, a=0, nonlinearity='relu'):
    """
    Initialize a module with kaiming normal init

    Args:
        m (nn.Module): the module to init
        a (float): the slope of the nonlinearity
        nonlinearity (str): type of the nonlinearity

    Returns:
        the initialized module
    """
    if nonlinearity in ['relu', 'leaky_relu']:
        if a == 0:
            nonlinearity = 'relu'
        else:
            nonlinearity = 'leaky_relu'

    nn.init.kaiming_normal_(m.weight, a=a, nonlinearity=nonlinearity)
    if hasattr(m, 'biais') and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return m


def xavier(m):
    """
    Initialize a module with xavier normal init

    Args:
        m (nn.Module): the module to init

    Returns:
        the initialized module
    """
    nn.init.xavier_normal_(m.weight)
    if hasattr(m, 'biais') and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return m


def n002(m):
    """
    Initialize a module with gaussian weights of standard deviation 0.02

    Args:
        m (nn.Module): the module to init

    Returns:
        the initialized module
    """
    nn.init.normal_(m.weight, 0, 0.02)
    if hasattr(m, 'biais') and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return m


def nb_parameters(net):
    """
    Counts the number of parameters of `net`

    Args:
        net (nn.Module): the net

    Returns:
        the number of params
    """
    return sum(p.numel() for p in net.parameters())


def layer_by_name(net, name):
    """
    Get a submodule at any depth of a net by its name

    Args:
        net (nn.Module): the base module containing other modules
        name (str): a name of a submodule of `net`, like `"layer3.0.conv1"`.

    Returns:
        The found layer or `None`
    """
    for l in net.named_modules():
        if l[0] == name:
            return l[1]


def forever(iterable):
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


def gram(m):
    """
    Return the Gram matrix of `m`

    Args:
        m (tensor): a matrix of dim 2

    Returns:
        The Gram matrix
    """
    m1 = m
    m2 = m.t()
    g = torch.mm(m1, m2) / m.shape[1]
    return g


def bgram(m):
    """
    Return the batched Gram matrix of `m`

    Args:
        m (tensor): a matrix of dim 3, first one is the batch

    Returns:
        The batch of Gram matrix
    """
    m = m.view(m.shape[0], m.shape[1], -1)
    m1 = m
    m2 = m.permute(0, 2, 1)
    g = torch.bmm(m1, m2) / (m.shape[1] * m.shape[2])
    return g


def dict_by_key(d, k):
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
    k = k.split('.')
    while len(k) != 0:
        if isinstance(d, dict):
            d = d[k[0]]
        else:
            d = d[int(k[0])]
        k = k[1:]
    return d


def send_to_device(x, device, non_blocking=False):
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


def recursive_state_dict(x):
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


def load_recursive_state_dict(x, obj):
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
        for k in objs.keys():
            load_recursive_state_dict(xx[k], oo[k])


class FrozenModule(nn.Module):
    """
    Wrap a module to eval model, can't be turned back to training mode

    Args:
        m (nn.Module): a module
    """
    def __init__(self, m):
        super(FrozenModule, self).__init__()
        self.m = freeze(m).eval()

    def train(self, mode=True):
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


def ilerp(a, b, t):
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
