import torchelie.utils as tu
from collections import OrderedDict


@tu.experimental
class Algorithm:
    """
    Define a customizable sequence of code blocks.
    """

    def __init__(self) -> None:
        self.passes = OrderedDict()

    def add_step(self, name: str, f=None):
        if f is not None:
            self.passes[name] = f
            return

        def _f(func):
            self.passes[name] = func
            return func

        return _f

    def __call__(self, *args, **kwargs):
        env = {}
        output = {}
        for pass_name, pass_ in self.passes.items():
            try:
                out = pass_(env, *args, **kwargs)
            except Exception as e:
                print('Error during pass', pass_name)
                raise e
            output.update(out)
        return output

    def remove_step(self, name: str):
        if name in self.passes:
            del self.passes[name]

    def insert_before(self, key: str, name: str, func=None):

        def _f(f):
            funs = list(self.passes.items())
            idx = [i for i, (k, v) in enumerate(funs) if k == key][0]
            funs[idx:idx] = [(name, f)]
            self.passes = OrderedDict(funs)
            return f

        if func is None:
            return _f
        else:
            _f(func)

    def insert_after(self, key: str, name: str, func=None):

        def _f(f):
            funs = list(self.passes.items())
            idx = [i for i, (k, v) in enumerate(funs) if k == key][0]
            funs[idx + 1:idx + 1] = [(name, f)]
            self.passes = OrderedDict(funs)
            return f

        if func is None:
            return _f
        else:
            _f(func)

    def __getitem__(self, name: str):
        return self.passes[name]

    def __setitem__(self, name: str, value):
        self.passes[name] = value

    def __repr__(self) -> str:
        return (self.__class__.__name__ + '\n' +
                tu.indent('\n'.join(list(self.passes.keys()))) + "\n")
