class ImageOptimizationBaseRecipe:
    def __init__(self, callbacks):
        self.state = {'metrics': {}}
        self.callbacks = callbacks

    def forward(self):
        raise NotImplemented

    def _run_cbs(self, trigger):
        for cb in self.callbacks:
            if hasattr(cb, trigger):
                getattr(cb, trigger)(self.state)

    def __call__(self, n_iters, *args, **kwargs):
        self.init(*args, **kwargs)
        self.iters = 0
        self._run_cbs('on_epoch_start')
        for i in range(n_iters):
            self._run_cbs('on_batch_start')
            out = self.forward()
            self.state.update(out)
            self._run_cbs('on_batch_end')
            self.iters += 1
        self._run_cbs('on_epoch_end')
        return self.result()
