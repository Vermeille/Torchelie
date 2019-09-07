import torch


def log_t(x, t):
    if t == 1:
        return torch.log(x)
    return (x**(1 - t) - 1) / (1 - t)


def exp_t(x, t):
    if t == 1:
        return torch.exp(x)
    return torch.clamp(1 + (1 - t) * x, min=0)**(1 / (1 - t))


def lambdas(a, t, n_iters=3):
    mu = torch.max(a, dim=1, keepdim=True).values
    a_tilde = a - mu
    for i in range(n_iters):
        za = exp_t(a_tilde, t).sum(1, keepdim=True)
        a_tilde = za**(1 - t) * (a - mu)
    return -log_t(1 / za, t) + mu


def tempered_softmax(x, t, n_iters=3):
    return exp_t(x - lambdas(x, t, n_iters=n_iters), t)


def tempered_cross_entropy(x, y, t1, t2, n_iters=3):
    """
    The bi-tempered loss from https://arxiv.org/abs/1906.03361

    Args:
        x (tensor): a tensor of batched probabilities like for cross_entropy
        y (tensor): a tensor of labels
        t1 (float): temperature 1
        t2 (float): temperature 2

    Returns:
        the loss
    """
    return tempered_nll_loss(tempered_softmax(x, t2, n_iters=n_iters), y, t1)


def tempered_nll_loss(x, y, t):
    y_hat = x[torch.arange(0, x.shape[0]).long(), y]
    return -log_t(y_hat, t) - (1 - torch.sum(x**(2 - t), dim=1)) / (2 - t)


class TemperedCrossEntropyLoss(torch.nn.Module):
    """
    The bi-tempered loss from https://arxiv.org/abs/1906.03361

    Args:
        t1 (float): temperature 1
        t2 (float): temperature 2
    """

    def __init__(self, t1, t2):
        super(TemperedCrossEntropy, self).__init__()
        self.t1 = t1
        self.t2 = t2

    def forward(self, x, y):
        """
        Forward pass

        Args:
            x (tensor): a tensor of batched probabilities like for
                cross_entropy
            y (tensor): a tensor of labels

        Returns:
            the loss
        """
        return tempered_cross_entropy(x, y, t1, t2)


if __name__ == '__main__':
    torch.manual_seed(0)
    a = torch.randn(12, 100)
    l = torch.arange(0, 12).long()

    print(tempered_softmax(a, 1))
    print(torch.nn.functional.softmax(a, dim=1))

    print(torch.nn.functional.cross_entropy(a, l, reduction='none'))
    print(tempered_cross_entropy(a, l, 1, 1))
