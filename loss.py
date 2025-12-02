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
    
class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances, targets):
        labels = torch.zeros_like(distances)
        labels.scatter_(1, targets.view(-1, 1), 1.0)
        pos = labels * torch.pow(distances, 2)
        neg = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        return torch.sum(pos + neg) * 0.5