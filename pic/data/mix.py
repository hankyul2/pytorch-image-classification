import math
import random

import torch
from torch.nn import functional as F


class MixUP:
    def __init__(self):
        self.p = 0.5
        self.alpha = 1.0

    def __call__(self, batch, target):
        if self.p > random.random():
            return batch, target

        ratio = 1 - torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]

        batch_roll = batch.roll(1, 0)
        target_roll = target.roll(1, 0)

        batch = batch * (1-ratio) + batch_roll * ratio
        target = F.one_hot(target) * (1-ratio) + F.one_hot(target_roll) * ratio

        return batch, target


class CutMix:
    def __init__(self):
        """CUTMIX
        Todo: add label smoothing option
        """
        self.p = 0.5
        self.alpha = 1.0

    def __call__(self, batch, target):
        if self.p > random.random():
            return batch, target

        ratio = 1 - torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]

        batch_roll = batch.roll(1, 0)
        target_roll = target.roll(1, 0)

        height_half = int(0.5 * math.sqrt(ratio) * batch.size(2))
        width_half = int(0.5 * math.sqrt(ratio) * batch.size(3))
        r = int(random.random() * batch.size(2))
        c = int(random.random() * batch.size(3))

        start_x = torch.clamp(r - height_half, min=0, max=batch.size(2))
        end_x = torch.clamp(r + height_half, min=0, max=batch.size(2))
        start_y = torch.clamp(r - width_half, min=0, max=batch.size(3))
        end_y = torch.clamp(r + width_half, min=0, max=batch.size(3))

        ratio = 1 - (end_x - start_x) * (end_y - start_y)

        batch = batch + batch_roll[:, :, start_x:end_x, start_y:end_y]
        target = F.one_hot(target) * (1-ratio) + F.one_hot(target_roll) * ratio

        return batch, target