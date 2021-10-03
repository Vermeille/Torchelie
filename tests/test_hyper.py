from torchelie.hyper import HyperparamSearch, UniformSampler


def beale(x, y):
    return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x +
                                                              x * y**3)**2


def sphere(x, y):
    return x**2 + y**2


def rosen(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2


hpsearch = HyperparamSearch(x=UniformSampler(-4.5, 4.5),
                            y=UniformSampler(-4.5, 4.5))

import os
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove('hpsearch.json')

print(beale(3, 0.5))
for _ in range(30):
    hps = hpsearch.sample(algorithm='gp', target='out')
    out = -beale(**hps.params)
    print(hps, '\t', out)
    hpsearch.log_result(hps, {'out': out})
