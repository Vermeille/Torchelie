import torch
import torch.nn.functional as F


def real(x):
    return F.relu(1 - x).mean()


def fake(x):
    return F.relu(1 + x).mean()


def generated(x):
    return -x.mean()
