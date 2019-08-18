class RunningAvg:
    def __init__(self):
        self.count = 0
        self.val = 0

    def log(self, x, total=1):
        self.count += total
        self.val += x

    def get(self):
        return self.val / self.count


class WindowAvg:
    def __init__(self, k=100):
        self.vals = []
        self.k = k

    def log(self, x):
        if len(self.vals) == self.k:
            self.vals = self.vals[1:]
        self.vals.append(x)

    def get(self):
        return sum(self.vals) / len(self.vals)


class ExponentialAvg:
    def __init__(self, beta=0.6):
        self.beta = beta
        self.val = None

    def log(self, x):
        if self.val is None:
            self.val = x
        else:
            self.val = self.beta * self.val + (1 - self.beta) * x

    def get(self):
        return self.val
