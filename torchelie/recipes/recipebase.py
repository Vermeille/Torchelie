import torchelie.metrics.callbacks as cb


class ImageOptimizationBaseRecipe:
    def __init__(self, visdom_env=None, log_every=10, callbacks=[]):
        self.callbacks = cb.CallbacksRunner(callbacks + [
            cb.WindowedMetricAvg('loss'),
            cb.VisdomLogger(visdom_env, log_every=log_every),
            cb.StdoutLogger(log_every=log_every),
        ])

    def forward(self):
        raise NotImplemented

    def init(self, *args, **kwargs):
        raise NotImplemented

    def __call__(self, n_iters, *args, **kwargs):
        state = {'metrics': {}, 'epoch': 0}

        self.init(*args, **kwargs)
        self.callbacks('on_epoch_start', state)
        for i in range(n_iters):
            state['iters'] = i
            self.iters = i
            state['epoch_batch'] = i
            self.callbacks('on_batch_start', state)

            out = self.forward()
            state.update(out)
            state['metrics']['img'] = self.result()

            self.callbacks('on_batch_end', state)

        self.callbacks('on_epoch_end', state)
        return self.result()
