import torch
from torchvision import transforms


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

        transform_list.extend([
            transforms.RandomResizedCrop(resize, interpolation=interpolation),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std)
        ])

        if remode:
            transform_list.append(transforms.RandomErasing(remode))

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)


class ImageNetVal:
    def __init__(self, test_resize, resize_mode, crop_ptr, interpolation, mean, std):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        if not isinstance(test_resize, (tuple, list)):
            test_resize = (test_resize, test_resize)

        crop_size = (int(test_resize[0] * crop_ptr), int(test_resize[1] * crop_ptr))

        if resize_mode == 'resize_shorter':
            test_resize = test_resize[0]

        transform_list = [
            transforms.Resize(test_resize, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean, std)
        ]

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)