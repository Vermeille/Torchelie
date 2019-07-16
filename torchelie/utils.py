import torch.nn as nn


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def entropy(out, dim, reduce='mean'):
    log_prob = F.log_softmax(out, dim=dim)
    h = -torch.sum(log_prob.exp() * log_prob, dim=dim)
    if reduce == 'none':
        return h
    if reduce == 'mean':
        return h.mean()
    if reduce == 'sum':
        return h.sum()


def kaiming(m, a=0, nonlinearity='relu'):
    if nonlinearity in ['relu', 'leaky_relu']:
        if a == 0:
            nonlinearity = 'relu'
        else:
            nonlinearity = 'leaky_relu'

    nn.init.kaiming_normal_(m.weight, a=a, nonlinearity=nonlinearity)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return m


def xavier(m):
    nn.init.xavier_normal_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return m
