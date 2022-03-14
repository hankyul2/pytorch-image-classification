import os

from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler, DataLoader
from torchvision.datasets import ImageFolder

from pic.data import MixUP, CutMix, RepeatAugSampler, ImageNetTrain, ImageNetVal


def get_dataset(args):
    train_dataset = ImageFolder(os.path.join(args.data_dir, 'train'), ImageNetTrain(args.train_resize, args.hflip, args.auto_aug, args.remode, args.interpolation, args.mean, args.std))
    val_dataset = ImageFolder(os.path.join(args.data_dir, 'val'), ImageNetVal(args.test_size, args.test_resize_mode, args.crop_ptr, args.interpolation, args.mean, args.std))
    args.num_classes = len(train_dataset.classes)

    return train_dataset, val_dataset


def get_dataloader(train_dataset, val_dataset, args):
    # 1. create sampler
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

    # 2. create collate_fn
    args.use_mixup = args.mixup or args.cutmix
    if args.mixup:
        # collate_fn.append(MixUP())
        collate_fn = MixUP()
    elif args.cutmix:
        # collate_fn.append(CutMix())
        collate_fn = CutMix()

    # 3. create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
                                  num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=args.pin_memory)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
                                  num_workers=args.num_workers, collate_fn=None, pin_memory=False)

    return train_dataloader, val_dataloader
