import torch


def log_t(x, t):
    """
    Logarithm.

    Args:
        x: (todo): write your description
        t: (todo): write your description
    """
    if t == 1:
        return torch.log(x)
    return (x**(1 - t) - 1) / (1 - t)


def exp_t(x, t):
    """
    Exp_t ( x t )

    Args:
        x: (todo): write your description
        t: (todo): write your description
    """
    if t == 1:
        return torch.exp(x)
    return torch.clamp(1 + (1 - t) * x, min=0)**(1 / (1 - t))


def lambdas(a, t, n_iters=3):
    """
    Compute the n_t ( n_t )

    Args:
        a: (array): write your description
        t: (array): write your description
        n_iters: (int): write your description
    """
    mu = torch.max(a, dim=1, keepdim=True).values
    a_tilde = a - mu
    for i in range(n_iters):
        za = exp_t(a_tilde, t).sum(1, keepdim=True)
        a_tilde = za**(1 - t) * (a - mu)
    return -log_t(1 / za, t) + mu


def tempered_log_softmax(x, t, n_iters=3):
    """
    Tempered log softmax. Computes log softmax along dimension 1

    Args:
        x (tensor): activations
        t (float): temperature
        n_iters (int): number of iters to converge (default: 3

    Returns:
        result of tempered log softmax
    """
    return x - lambdas(x, t, n_iters=n_iters)


def tempered_softmax(x, t, n_iters=3):
    """
    Tempered softmax. Computes softmax along dimension 1

    Args:
        x (tensor): activations
        t (float): temperature
        n_iters (int): number of iters to converge (default: 3

    Returns:
        result of tempered softmax
    """
    return exp_t(tempered_log_softmax(x, t, n_iters), t)


def tempered_cross_entropy(x,
                           y,
                           t1,
                           t2,
                           n_iters=3,
                           weight=None,
                           reduction='mean'):
    """
    The bi-tempered loss from https://arxiv.org/abs/1906.03361

    Args:
        x (tensor): a tensor of batched probabilities like for cross_entropy
        y (tensor): a tensor of labels
        t1 (float): temperature 1
        t2 (float): temperature 2
        weight (tensor): a tensor that associates a weight to each class
        reduction (str): how to reduce the batch of losses: 'none', 'sum', or
            'mean'

    Returns:
        the loss
    """
    sm = tempered_log_softmax(x, t2, n_iters=n_iters)
    return tempered_nll_loss(sm, y, t1, t2, weight=weight, reduction=reduction)


def tempered_nll_loss(x, y, t1, t2, weight=None, reduction='mean'):
    """
    Compute tempered nll loss

    Args:
        x (tensor): activations of log softmax
        y (tensor): labels
        t1 (float): temperature 1
        t2 (float): temperature 2
        weight (tensor): a tensor that associates a weight to each class
        reduction (str): how to reduce the batch of losses: 'none', 'sum', or
            'mean'
    Returns:
        the loss
    """
    x = exp_t(x, t2)
    y_hat = x[torch.arange(0, x.shape[0]).long(), y]
    out = -log_t(y_hat, t1) - (1 - torch.sum(x**(2 - t1), dim=1)) / (2 - t1)
    if weight is not None:
        out = weight[y] * out

    if reduction == 'none':
        return out
    if reduction == 'mean' and weight is not None:
        return torch.sum(out / weight[y].sum())
    if reduction == 'mean' and weight is None:
        return torch.sum(out / out.shape[0])
    if reduction == 'sum':
        return out.sum()


class TemperedCrossEntropyLoss(torch.nn.Module):
    """
    The bi-tempered loss from https://arxiv.org/abs/1906.03361

    Args:
        t1 (float): temperature 1
        t2 (float): temperature 2
        weight (tensor): a tensor that associates a weight to each class
        reduction (str): how to reduce the batch of losses: 'none', 'sum', or
            'mean'
    """

    def __init__(self, t1, t2, weight=None, reduction='mean'):
        """
        Initialize the gradient

        Args:
            self: (todo): write your description
            t1: (int): write your description
            t2: (int): write your description
            weight: (int): write your description
            reduction: (todo): write your description
        """
        super(TemperedCrossEntropyLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.weight = weight
        self.reduction = reduction

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
        return tempered_cross_entropy(x,
                                      y,
                                      self.t1,
                                      self.t2,
                                      weight=self.weight,
                                      reduction=self.reduction)


if __name__ == '__main__':
    torch.manual_seed(0)
    a = torch.randn(12, 100)
    l = torch.arange(0, 12).long()
    w = torch.randn(100).abs() * 10

    print(tempered_softmax(a, 1))
    print(torch.nn.functional.softmax(a, dim=1))

    print(torch.nn.functional.cross_entropy(a, l, reduction='mean', weight=w))
    print(torch.nn.functional.cross_entropy(a, l, reduction='sum', weight=w))
    print(tempered_cross_entropy(a, l, 1, 1, weight=w, reduction='mean'))
