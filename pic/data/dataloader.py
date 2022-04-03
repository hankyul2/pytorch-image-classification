from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler, DataLoader

from pic.data import MixUP, CutMix, RepeatAugSampler


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

    args.iter_per_epoch = len(train_dataloader)

    return train_dataloader, val_dataloader
