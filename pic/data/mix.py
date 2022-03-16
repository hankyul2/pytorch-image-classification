import math
import random
import numpy as np

import torch
from torch.nn import functional as F


def one_hot_label_smoothing(y, nclass, smooth, confidence, device):
    return torch.full((y.size(0), nclass), smooth, device=device).scatter_(1, y.unsqueeze(1), confidence)


def mix_target(y1, y2, ratio, nclass, label_smoothing=0.0, device='cuda'):
    smooth = label_smoothing / nclass
    confidence = 1 - label_smoothing + smooth
    y1 = one_hot_label_smoothing(y1, nclass, smooth, confidence, device=device)
    y2 = one_hot_label_smoothing(y2, nclass, smooth, confidence, device=device)
    return y1 * ratio + y2 * (1 - ratio)


def get_cutmix_bounding_box_rectangle(img_shape, min_max_ratio, size=None):
    H, W = img_shape[:2]
    min_ratio, max_ratio = min_max_ratio
    bb_h = np.random.randint(int(min_ratio * H), int(max_ratio * H), size=size)
    bb_w = np.random.randint(int(min_ratio * W), int(max_ratio * W), size=size)
    start_h = np.random.randint(0, H - bb_h, size=size)
    start_w = np.random.randint(0, W - bb_w, size=size)
    return start_w, start_w + bb_w, start_h, start_h + bb_h


def get_cutmix_bounding_box(img_shape, ratio):
    # Todo: improve me
    """get bounding box

    control bounding box shape (rectangle, square)
    control bounding box clip (clip, not clip)
    """
    if not isinstance(ratio, (list, tuple)):
        ratio = (0, ratio)

    x_s, x_e, y_s, y_e = get_cutmix_bounding_box_rectangle(img_shape, ratio)
    ratio = (x_e - x_s) * (y_e - y_s) / (img_shape[0] * img_shape[1])

    return (x_s, x_e, y_s, y_e), ratio


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