import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryCrossEntropy(nn.Module):

    def __init__(self, reduction="mean", weighted_loss=False):
        super(BinaryCrossEntropy, self).__init__()

        self.reduction = reduction
        self.weighted_loss = weighted_loss
 
    def forward(self, inputs, targets):
        labels = torch.zeros_like(inputs)
        labels.scatter_(1, targets.view(-1, 1), 1.0)
        weights = (1 + 6 * labels) / 7 if self.weighted_loss else None
        return F.binary_cross_entropy_with_logits(inputs, labels, weight=weights, reduction=self.reduction)