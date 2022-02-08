import os

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


def train_transform(resize, hflip, auto_aug, remode, interpolation, mean, std):
    transform_list = []

    if hflip:
        transform_list.append(transforms.RandomHorizontalFlip(hflip))

    if auto_aug:
        if auto_aug.starts_with('ra'):
            transform_list.append(transforms.RandAugment(interpolation=interpolation))
        elif auto_aug.starts_with('ta_wide'):
            transform_list.append(transforms.TrivialAugmentWide(interpolation=interpolation))
        elif auto_aug.starts_with('aa'):
            policy = transforms.AutoAugmentPolicy('imagenet')
            transform_list.append(transforms.AutoAugment(policy=policy, interpolation=interpolation))

    if remode:
        transform_list.append(transforms.RandomErasing(remode))

    transform_list.extend([
        transforms.RandomResizedCrop(resize, interpolation=interpolation),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std)
    ])

    transform_fn = transforms.Compose(transform_list)

    return transform_fn


def val_transform(test_resize, crop_ptr, interpolation, mean, std):
    transform_list = [
        transforms.Resize(test_resize, interpolation=interpolation),
        transforms.CenterCrop(test_resize),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean, std)
    ]

    transform_fn = transforms.Compose(transform_list)

    return transform_fn


def get_data(args):
    interpolation = transforms.functional.InterpolationMode(args.interpolation)
    train_dataset = ImageFolder(os.path.join(args.data_dir, 'train'), train_transform(args.train_resize, args.hflip, args.auto_aug, args.remode, interpolation, args.mean, args.std))
    test_dataset = ImageFolder(os.path.join(args.data_dir, 'val'), val_transform(args.test_resize, args.crop_ptr, interpolation, args.mean, args.std))

    return None, None
