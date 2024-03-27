import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys

def symmetric_cross_entropy_loss(logits):
    # symmetric loss function

    # b * b
    # im ev [b dim] * [dim b]

    if logits == []:
        return torch.tensor(0, dtype=torch.long)

    device = logits.device
    num_gpu = int(logits.shape[0] / logits.shape[1])
    batch_size = logits.shape[1]

    if logits.shape[0] != logits.shape[1]:
        loss = 0.0
        for i in range(num_gpu):
            logits_per_gpu = logits[batch_size * i:batch_size * (i + 1), :]
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            loss_i = F.cross_entropy(logits_per_gpu, labels)
            loss_t = F.cross_entropy(logits_per_gpu.T, labels)
            loss += (loss_i + loss_t) / 2
    else:
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        loss = (loss_i + loss_t) / 2

    return loss
