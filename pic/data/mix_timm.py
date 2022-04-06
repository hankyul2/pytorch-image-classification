import numpy as np
import torch


def one_hot_label_smoothing(y, nclass, smooth, confidence, device):
    return torch.full((y.size(0), nclass), smooth, device=device).scatter_(1, y.unsqueeze(1), confidence)


def mix_target(y1, y2, ratio, nclass, label_smoothing=0.0, device='cuda'):
    smooth = label_smoothing / nclass
    confidence = 1 - label_smoothing + smooth
    y1 = one_hot_label_smoothing(y1, nclass, smooth, confidence, device=device)
    y2 = one_hot_label_smoothing(y2, nclass, smooth, confidence, device=device)
    return y1 * ratio + y2 * (1 - ratio)


def get_cutmix_bounding_box_square(img_shape, ratio, margin=0.0, size=None):
    H, W = img_shape[-2:]
    ratio = np.sqrt(1-ratio)
    bb_h, bb_w = int(H * ratio), int(W * ratio)
    margin_h, margin_w = int(bb_h * margin), int(bb_w * margin)
    center_h = np.random.randint(0 + margin_h, H - margin_h, size=size)
    center_w = np.random.randint(0 + margin_w, W - margin_w, size=size)
    start_h = np.clip(center_h - bb_h // 2, 0, H)
    end_h = np.clip(center_h + bb_h // 2, 0, H)
    start_w = np.clip(center_w - bb_w // 2, 0, W)
    end_w = np.clip(center_w + bb_w // 2, 0, W)
    return start_h, end_h, start_w, end_w


def get_cutmix_bounding_box_rectangle(img_shape, min_max_ratio, size=None):
    H, W = img_shape[-2:]
    min_ratio, max_ratio = min_max_ratio
    bb_h = np.random.randint(int(min_ratio * H), int(max_ratio * H), size=size)
    bb_w = np.random.randint(int(min_ratio * W), int(max_ratio * W), size=size)
    start_h = np.random.randint(0, H - bb_h, size=size)
    start_w = np.random.randint(0, W - bb_w, size=size)
    return start_h, start_h + bb_h, start_w, start_w + bb_w


def get_cutmix_bbox_and_ratio(img_shape, lam, correct_lam=True, size=None):
    if isinstance(lam, (tuple, list)):
        yl, yu, xl, xu = get_cutmix_bounding_box_rectangle(img_shape, lam, size=size)
    else:
        yl, yu, xl, xu = get_cutmix_bounding_box_square(img_shape, lam, size=size)
    if correct_lam:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class Mixup_Cutmix_Smoothing:
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, label_smoothing=0.1, nclass=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.nclass = nclass
        self.enable_mixup = True

    def choose_between_cutmix_mixup_per_batch(self):
        ratio = 1.0

        if np.random.rand() < self.prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                ratio = self.get_cutmix_ratio() if use_cutmix else self.get_cutmix_ratio()
            elif self.mixup_alpha > 0.:
                use_cutmix = False
                ratio = self.get_mixup_ratio()
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                ratio = self.get_cutmix_ratio()
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."

        return ratio, use_cutmix

    def get_mixup_ratio(self):
        return float(np.random.beta(self.mixup_alpha, self.mixup_alpha))

    def get_cutmix_ratio(self):
        return float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))

    def mix_batch(self, output, batch):
        batch_size = len(batch)
        ratio, use_cutmix = self.choose_between_cutmix_mixup_per_batch()
        if use_cutmix:
            (yl, yh, xl, xh), lam = get_cutmix_bbox_and_ratio(output.shape, ratio, self.correct_lam)
        for i in range(batch_size):
            j = batch_size - i - 1
            mixed = batch[i][0]
            if lam != 1.:
                if use_cutmix:
                    mixed = mixed.copy()  # don't want to modify the original while iterating
                    mixed[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh]
                else:
                    mixed = mixed.astype(np.float32) * lam + batch[j][0].astype(np.float32) * (1 - lam)
                    np.rint(mixed, out=mixed)
            output[i] += torch.from_numpy(mixed.astype(np.uint8))
        return lam