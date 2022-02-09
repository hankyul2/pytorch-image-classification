import math
import os
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler, Sampler, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageNetTrain:
    def __init__(self, resize, hflip, auto_aug, remode, interpolation, mean, std):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        transform_list = []

        if hflip:
            transform_list.append(transforms.RandomHorizontalFlip(hflip))

        if auto_aug:
            if auto_aug.startswith('ra'):
                transform_list.append(transforms.RandAugment(interpolation=interpolation))
            elif auto_aug.startswith('ta_wide'):
                transform_list.append(transforms.TrivialAugmentWide(interpolation=interpolation))
            elif auto_aug.startswith('aa'):
                policy = transforms.AutoAugmentPolicy('imagenet')
                transform_list.append(transforms.AutoAugment(policy=policy, interpolation=interpolation))

        if remode:
            transform_list.append(transforms.RandomErasing(remode))

        transform_list.extend([
            transforms.RandomResizedCrop(resize, interpolation=interpolation),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)


class ImageNetVal:
    def __init__(self, test_resize, crop_ptr, interpolation, mean, std):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        transform_list = [
            transforms.Resize(test_resize, interpolation=interpolation),
            transforms.CenterCrop(test_resize),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean, std)
        ]

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)


class RepeatAugSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU). Heavily based on torch.utils.data.DistributedSampler
    This sampler was taken from https://github.com/facebookresearch/deit/blob/0c4b8f60/samplers.py
    Used in
    Copyright (c) 2015-present, Facebook, Inc.
    """

    def __init__(
            self,
            dataset,
            num_replicas=None,
            rank=None,
            shuffle=True,
            num_repeats=3,
            selected_round=256,
            selected_ratio=0,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # Determine the number of samples to select per epoch for each rank.
        # num_selected logic defaults to be the same as original RASampler impl, but this one can be tweaked
        # via selected_ratio and selected_round args.
        selected_ratio = selected_ratio or num_replicas  # ratio to reduce selected samples by, num_replicas if 0
        if selected_round:
            self.num_selected_samples = int(math.floor(
                 len(self.dataset) // selected_round * selected_round / selected_ratio))
        else:
            self.num_selected_samples = int(math.ceil(len(self.dataset) / selected_ratio))

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # produce repeats e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2....]
        indices = torch.repeat_interleave(indices, repeats=self.num_repeats, dim=0).tolist()
        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample per rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # return up to num selected samples
        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


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


def get_data(args):
    train_dataset = ImageFolder(os.path.join(args.data_dir, 'train'), ImageNetTrain(args.train_resize, args.hflip, args.auto_aug, args.remode, args.interpolation, args.mean, args.std))
    val_dataset = ImageFolder(os.path.join(args.data_dir, 'val'), ImageNetVal(args.test_resize, args.crop_ptr, args.interpolation, args.mean, args.std))

    if args.distributed:
        if args.aug_repeat:
            train_sampler = RepeatAugSampler(train_dataset, num_repeats=args.aug_repeat)
        else:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    collate_fn = None

    if args.mixup:
        # collate_fn.append(MixUP())
        collate_fn = MixUP()
    elif args.cutmix:
        # collate_fn.append(CutMix())
        collate_fn = CutMix()

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
                                  num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=args.pin_memory)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
                                  num_workers=args.num_workers, collate_fn=None, pin_memory=False)

    return train_dataloader, val_dataloader
