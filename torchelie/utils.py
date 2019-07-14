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

