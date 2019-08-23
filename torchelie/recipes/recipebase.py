import torchelie.metrics.callbacks as cb


class ImageOptimizationBaseRecipe:
    def __init__(self, visdom_env=None, log_every=10, callbacks=[]):
        self.callbacks = cb.CallbacksRunner(callbacks + [
            cb.WindowedMetricAvg('loss'),
            cb.VisdomLogger(visdom_env, log_every=10),
            cb.StdoutLogger(log_every=10),
        ])

    def forward(self):
        raise NotImplemented

    def init(self, *args, **kwargs):
        raise NotImplemented

    def __call__(self, n_iters, *args, **kwargs):
        state = {'metrics': {}}

        self.init(*args, **kwargs)
        self.iters = 0
        self.callbacks('on_epoch_start', state)
        for i in range(n_iters):
            self.callbacks('on_batch_start', state)

            out = self.forward()
            self.state.update(out)
            self.state['metrics']['img'] = self.result()

            self.callbacks('on_batch_end', state)

            self.iters += 1

        self.callbacks('on_epoch_end', state)
        return self.result()
