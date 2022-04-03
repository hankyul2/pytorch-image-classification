import pytest
from collections import namedtuple
from pic.data import get_dataset


@pytest.fixture
def args():
    args = namedtuple('args', 'data_dir classes dataset_type train_resize train_resize_mode train_crop_pad hflip auto_aug remode interpolation mean std test_size test_resize_mode crop_ptr')
    args.data_dir = 'data'
    args.dataset_type = 'FashionMNIST'
    args.train_resize = 224
    args.train_resize_mode = 'ResizeRandomCrop'
    args.train_crop_pad = 4
    args.interpolation = 'bilinear'
    args.mean = [0.5]
    args.std = [0.5]
    args.test_size = 224
    args.test_resize_mode = 'resize_shorter'
    args.crop_ptr = 0.9
    args.auto_aug = 'ra'
    args.remode = 0.2
    return args


def test_get_dataset(args):
    train_dataset, val_dataset = get_dataset(args)
