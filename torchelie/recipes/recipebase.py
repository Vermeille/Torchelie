import torch
from collections import defaultdict, OrderedDict
import torchelie.utils as tu


class CallbacksRunner:
    def __init__(self):
        self.cbs = [[], [], []]
        self.reset()

    def reset(self):
        self.state = {'metrics': {}}

    def __call__(self, name, *args, **kwargs):
        for cb in self.callbacks():
            if hasattr(cb, name):
                getattr(cb, name)(self.state, *args, **kwargs)

    def callbacks(self):
        for cbs in self.cbs:
            for cb in cbs:
                yield cb

    def named_callbacks(self):
        for step, cbs in zip(['prologue', 'middle', 'epilogue'], self.cbs):
            counts = defaultdict(int)
            for cb in cbs:
                nm = cb.__class__.__name__
                cnt = counts[nm]
                counts[nm] += 1
                yield '_'.join([nm, step, str(cnt)]), cb

    def state_dict(self):
        serial_cb = {}
        for nm, cb in self.named_callbacks():
            if hasattr(cb, 'state_dict'):
                serial_cb[nm] = cb.state_dict()
        return {'state': self.state, 'callbacks': serial_cb}

    def load_state_dict(self, dicc):
        self.state = dicc['state']

        for nm, cb in self.named_callbacks():
            if hasattr(cb, 'load_state_dict') and nm in dicc['callbacks']:
                cb.load_state_dict(dicc['callbacks'][nm])

    def update_state(self, state_additions):
        self.state.update(state_additions)

    def add_prologue(self, cb):
        self.cbs[0].append(cb)

    def add_callback(self, cb):
        self.cbs[1].append(cb)

    def add_epilogue(self, cb):
        self.cbs[2].append(cb)

    def add_prologues(self, cbs):
        for cb in cbs:
            self.add_prologue(cb)

    def add_callbacks(self, cbs):
        for cb in cbs:
            self.add_callback(cb)

    def add_epilogues(self, cbs):
        for cb in cbs:
            self.add_epilogue(cb)

    def __repr__(self):
        return "Prologue:\n{}\nCallbacks:\n{}\nEpilogue:\n{}".format(
            "\n".join(["  " + line for line in repr(self.cbs[0]).split("\n")]),
            "\n".join(["  " + line for line in repr(self.cbs[1]).split("\n")]),
            "\n".join(["  " + line for line in repr(self.cbs[2]).split("\n")]),
        )


class RecipeBase:
    def __init__(self):
        self._modules = set()
        self._savable = set()
        self.device = 'cpu'

    def _check_init(self):
        if '_modules' not in self.__dict__:
            raise AttributeError('You forgot to call ModulesAware.__init__()')

    def register(self, name, value):
        """
        Register an object into the recipe as a member. Calling
        :code:`recipe.register('foo', bar)` registers bar, and makes it usable
        through :code:`recipe.foo`.

        Args:
            name (str): member's name
            value: the object to register
        """
        self._check_init()
        self._modules.discard(name)
        self._savable.discard(name)

        if isinstance(value, torch.nn.Module):
            self._modules.add(name)
        else:
            self._savable.add(name)

        self.__dict__[name] = value

    def state_dict(self):
        """
        Returns:
            A state dict
        """
        sd = OrderedDict()
        for nm in self._modules:
            mod = self.__dict__[nm]
            if isinstance(mod, (torch.nn.parallel.DistributedDataParallel,
                                torch.nn.parallel.DataParallel)):
                sd[nm] = mod.module.state_dict()
            else:
                sd[nm] = mod.state_dict()

        for nm in self._savable:
            val = self.__dict__[nm]
            if hasattr(val, 'state_dict'):
                sd[nm] = val.state_dict()
            else:
                sd[nm] = val
        return sd

    def load_state_dict(self, state_dict):
        """
        Restore a recipe
        """
        for key, state in state_dict.items():
            val = self.__dict__[key]
            if hasattr(val, 'load_state_dict'):
                if isinstance(val, torch.nn.Module):
                    if isinstance(val,
                                  (torch.nn.parallel.DistributedDataParallel,
                                   torch.nn.parallel.DataParallel)):
                        print(val.module.load_state_dict(state, strict=False))
                    else:
                        print(val.load_state_dict(state, strict=False))
                else:
                    val.load_state_dict(state)
            else:
                self.__dict__[key] = val
        return self

    def modules(self):
        """
        Iterate over all nn.Modules registered in the recipe
        """
        for m in self._modules:
            yield self.__dict__[m]

    def to(self, device):
        """
        Move a recipe and all its movable registered objects to a device

        Args:
            device: a torch device
        """
        self._check_init()
        self.device = device

        for m in self.modules():
            m.to(self.device)

        for nm in self._savable:
            m = self.__dict__[nm]
            if hasattr(m, 'to'):
                self.__dict__[nm] = m.to(self.device)

        return self

    def cuda(self):
        """
        Move a recipe and all its movable registered objects to cuda
        """
        return self.to('cuda')

    def cpu(self):
        """
        Move a recipe and all its movable registered objects to cpu
        """
        return self.to('cpu')


class Recipe(RecipeBase):
    """
    Basic recipe that iterates mutiple epochs over a dataset. That loop is
    instrumented through several configurable callbacks. Callbacks can handle
    events before and after each batch, before and after each epoch. Each batch
    is treated with a user supplied function that manipulates it and returns a
    dict that returns a state usable by the callbacks.

    A recipe can be saved by calling its :code:`state_dict()` member. All its
    hyper parameters and state will be saved so that it can restart, but
    requires the exact same setting of callbacks.

    You can register multiple objects to a Recipe with :code:`register()`. If
    it has a :code:`state_dict()` member, its state will be saved into the
    recipe's when calling the recipe's :code:`state_dict()` function. If it
    has a member :code:`to()`, moving the recipe to another device will also
    move those objects.

    Args:
        call_fun (Callable): A function that takes a batch as an argument and
            returns a dict of value to feed the state
        loader (Iterable): any iterable (most likely a DataLoader)
    """
    def __init__(self, call_fun, loader):
        super(Recipe, self).__init__()
        self.call_fun = call_fun
        self.loader = loader

        self.callbacks = CallbacksRunner()
        self.register('callbacks', self.callbacks)

    def run(self, epochs):
        """
        Run the recipe for :code:`epochs` epochs.

        Args:
            epochs (int): number of epochs

        Returns:
            The state
        """
        self.to(self.device)
        for epoch in range(epochs):
            self.callbacks('on_epoch_start')
            for batch in self.loader:
                self.callbacks.update_state({'batch': batch})
                batch = tu.send_to_device(batch,
                                          self.device,
                                          non_blocking=True)
                self.callbacks.update_state({'batch_gpu': batch})

                self.callbacks('on_batch_start')
                out = self.call_fun(batch)
                out = tu.send_to_device(out, 'cpu', non_blocking=False)
                self.callbacks.update_state(out)
                self.callbacks('on_batch_end')
            self.callbacks('on_epoch_end')
        return self.callbacks.state
