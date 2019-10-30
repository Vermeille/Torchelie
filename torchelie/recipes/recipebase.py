from collections import defaultdict
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
            if hasattr(cb, 'load_state_dict'):
                cb.load_state_dict(dicc['callbacks'][nm])

    def update_state(self, state_additions):
        self.state.update(state_additions)

    def add_prologue(self, cb):
        self.cbs[0].append(cb)

    def add_callback(self, cb):
        self.cbs[1].append(cb)

    def add_epilogue(self, cb):
        self.cbs[2].append(cb)


class DataLoop:
    def __init__(self, call_fun, loader, device='cpu'):
        self.device = device
        self.call_fun = call_fun
        self.callbacks = CallbacksRunner()
        self.loader = loader

    def state_dict(self):
        return {'callbacks': self.callbacks.state_dict()}

    def add_prologues(self, cbs):
        for cb in cbs:
            self.callbacks.add_prologue(cb)
        return self

    def add_callbacks(self, cbs):
        for cb in cbs:
            self.callbacks.add_callback(cb)
        return self

    def add_epilogues(self, cbs):
        for cb in cbs:
            self.callbacks.add_epilogue(cb)
        return self

    def set_initial_state(self, state):
        self.callbacks.update_state(state)
        return self

    def run(self, epochs):
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
                out = tu.send_to_device(out, 'cpu', non_blocking=True)
                self.callbacks.update_state(out)
                self.callbacks('on_batch_end')
            self.callbacks('on_epoch_end')
        return self.callbacks.state


class DataModelLoop(DataLoop):
    def __init__(self, model, call_fun, loader, device='cpu'):
        super(DataModelLoop, self).__init__(call_fun, loader, device)
        self.model = model

    def state_dict(self):
        dicc = super(DataModelLoop, self).state_dict()
        dicc['model'] = self.model.state_dict()
        return dicc

    def load_state_dict(self, dicc):
        super(DataModelLoop, self).load_state_dict(dicc['callbacks'])
        self.model.load_state_dict(dicc['model'])

    def run(self, epochs):
        self.model.to(self.device)
        return super(DataModelLoop, self).run(epochs)
