import os

from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, FashionMNIST

from pic.data import TrainTransform, ValTransform, MyCIFAR100, MyCaltech101, MyCUB200, MyMITIndoor, MyTinyImageNet200


dataset_dict = {
    'ImageFolder': ImageFolder,
    'CIFAR10': CIFAR10,
    'CIFAR100': CIFAR100,
    'MyCIFAR100': MyCIFAR100,
    'FashionMNIST': FashionMNIST,
    'Caltech101': MyCaltech101,
    'CUB200': MyCUB200,
    'MITIndoor': MyMITIndoor,
    'TinyImageNet': MyTinyImageNet200,
}


def get_dataset(args):
    dataset_class = dataset_dict[args.dataset_type]
    train_transform = TrainTransform(args.train_size, args.train_resize_mode, args.random_crop_pad, args.random_crop_scale, args.random_crop_ratio, args.hflip, args.auto_aug, args.remode, args.interpolation, args.mean, args.std)
    val_transform = ValTransform(args.test_size, args.test_resize_mode, args.center_crop_ptr, args.interpolation, args.mean, args.std)
    if args.dataset_type == 'ImageFolder':
        train_dataset = dataset_class(os.path.join(args.data_dir, args.train_split), train_transform)
        val_dataset = dataset_class(os.path.join(args.data_dir, args.val_split), val_transform)
        args.num_classes = len(train_dataset.classes)
    elif args.dataset_type in ['CIFAR10', 'CIFAR100', 'FashionMNIST', 'Caltech101', 'CUB200', 'MITIndoor', 'TinyImageNet']:
        train_dataset = dataset_class(root=args.data_dir, train=True, download=True, transform=train_transform)
        val_dataset = dataset_class(root=args.data_dir, train=False, download=True, transform=val_transform)
        args.num_classes = len(train_dataset.classes)
    # elif args.dataset_type in ['MyCIFAR100']:
    #     train_dataset = dataset_class(args.data_dir, train=True, download=True, size=args.train_size, pad=args.random_crop_pad, interpolation=args.interpolation, remode=args.remode)
    #     val_dataset = dataset_class(args.data_dir, train=False, download=True, size=args.test_size, interpolation=args.interpolation)
    #     args.num_classes = len(train_dataset.classes)
    else:
        assert f"{args.dataset_type} is not supported yet. Just make your own code for it"

    return train_dataset, val_dataset