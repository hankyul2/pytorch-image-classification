import torch
from torch import nn
from torch.nn import functional as F


class BinaryCrossEntropy(nn.Module):
    def __init__(self, label_smoothing=0.1, bce_target=None):
        """Binary Cross Entropy (timm)
        :arg
            label_smoothing: multi-class loss in bce.
            bce_target: remove uncertain target used with cutmix.
        """
        super(BinaryCrossEntropy, self).__init__()
        self.smoothing = label_smoothing
        self.bce_target = bce_target

    def forward(self, x, y):
        if x.shape != y.shape:
            smooth = self.smoothing / x.size(-1)
            label = 1.0 - self.smoothing + smooth
            smooth_target = torch.full_like(x, smooth)
            y = torch.scatter(smooth_target, -1, y.long().view(-1, 1), label)
        if self.bce_target:
            y = torch.gt(y, self.bce_target).long()
        return F.binary_cross_entropy_with_logits(x, y, reduction='mean')