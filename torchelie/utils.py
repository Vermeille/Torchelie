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

def n002(m):
    nn.init.normal_(m.weight, 0, 0.02)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return m


def nb_parameters(net):
    return sum(p.numel() for p in net.parameters())


def layer_by_name(net, name):
    for l in net.named_modules():
        if l[0] == name:
            return l[1]

def forever(iterable):
    it = iter(iterable)
    while True:
        try:
            yield next(it)
        except Exception as e:
            print(e)
            it = iter(iterable)


def gram(m):
    m1 = m
    m2 = m.t()
    g = torch.mm(m1, m2) / m.shape[1]
    return g


def bgram(m):
    m = m.view(m.shape[0], m.shape[1], -1)
    m1 = m
    m2 = m.permute(0, 2, 1)
    g = torch.bmm(m1, m2) / (m.shape[1] * m.shape[2])
    return g
