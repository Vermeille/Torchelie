import math
import random
import json
import copy


class Sampler:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return random.uniform(self.low, self.high)

    def inverse(self, x):
        return x


class ExpSampler(Sampler):
    def __init__(self, low, high):
        low = self.inverse(low)
        high = self.inverse(high)
        super(ExpSampler, self).__init__(low, high)

    def sample(self):
        return 10**super(ExpSampler, self).sample()

    def inverse(self, x):
        return math.log10(x)


class DecaySampler(ExpSampler):
    def __init__(self, low, high):
        super(DecaySampler, self).__init__(low, high)

    def sample(self):
        return 1 - super(DecaySampler, self).sample()

    def inverse(self, x):
        return super(DecaySampler, self).inverse(1 - x)


class ChoiceSampler:
    def __init__(self, choices):
        self.choices = choices

    def sample(self):
        return random.choice(self.choices)

    def inverse(self, x):
        return self.choices.index(x)


class HyperparamSampler:
    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams

    def sample(self):
        return {k: v.sample() for k, v in self.hyperparams.items()}


class HyperparamSearch:
    def __init__(self, **hyperparams):
        self.sampler = HyperparamSampler(**hyperparams)

    def sample(self):
        return self.sampler.sample()

    def read_hpsearch(self):
        try:
            with open('hpsearch.json', 'r') as f:
                return json.load(f)
        except:
            return []

    def _str(self, x):
        try:
            return float(x)
        except:
            pass

        try:
            return x.item()
        except:
            pass

        return x

    def log_result(self, hps, result):
        res = self.read_hpsearch()
        full = copy.deepcopy(res)
        full.update({'result_' + k: self._str(v) for k, v in result.items()})
        res.append(full)
        with open('hpsearch.json', 'w') as f:
            json.dump(res, f)
