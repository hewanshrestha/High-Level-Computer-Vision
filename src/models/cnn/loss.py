import torch.nn.functional as F


def cce_loss(output, target):
    return F.cross_entropy(output, target)
