from typing import Tuple, Any

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR100


class MyCIFAR100(CIFAR100):
    def __init__(self, *args, size=224, pad=28, interpolation='bicubic', remode=0.5, **kwargs):
        super(MyCIFAR100, self).__init__(*args, **kwargs)
        if kwargs['train']:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size, padding=pad),
                transforms.RandomErasing(remode)
            ])
        else:
            self.transform = None

        interpolation = transforms.functional.InterpolationMode(interpolation)
        dataset = torch.from_numpy(self.data).permute(0, 3, 1, 2)
        B, C, H, W = dataset.shape
        dataset = dataset.reshape(5, B // 5, C, H, W)
        preprocess_transforms = torch.nn.Sequential(
            transforms.Resize(224, interpolation=interpolation),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        )
        preprocessed_data = []
        for data in dataset:
            data = data.cuda()
            with torch.no_grad():
                data = preprocess_transforms(data).to('cpu')
                preprocessed_data.append(data)
        self.data = torch.cat(preprocessed_data, dim=0)
        torch.cuda.empty_cache()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

