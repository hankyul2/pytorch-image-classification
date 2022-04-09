import os
import zipfile
import gdown
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision.datasets import Caltech101, ImageFolder


def download_cub200(data_root='data'):
    url = 'https://drive.google.com/uc?id=1cxfLzjWRqhdkwAKzbx7I1C4RqmkYrAED'
    store_path = f'{data_root}/zip/cub200.zip'
    if os.path.exists(store_path):
        print('CUB200 data already downloaded')
    else:
        gdown.download(url, store_path, quiet=False)
        unzip(store_path, data_root)


def download_mit_indoor(data_root='data'):
    url = 'https://drive.google.com/uc?id=1bMtOszV4UUyH0Xf-Br110EWOX9-1iHv7'
    store_path = f'{data_root}/zip/mit_indoor.zip'
    if os.path.exists(store_path):
        print('MiT Indoor data already downloaded')
    else:
        gdown.download(url, store_path, quiet=False)
        unzip(store_path, data_root)


def download_tinyimagenet(data_root='data'):
    url = 'https://drive.google.com/uc?id=1JNKvfFLDOmbxfw0UvpJj9r_j4EIm7Zkl'
    store_path = f'{data_root}/zip/tiny_imagenet_200.zip'
    if os.path.exists(store_path):
        print('TinyImageNet data already downloaded')
    else:
        gdown.download(url, store_path, quiet=False)
        unzip(store_path, data_root)


def unzip(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


class MyCaltech101(Caltech101):
    def __init__(self, *args, train=True, **kwargs):
        super(MyCaltech101, self).__init__(*args, **kwargs)
        X_train, X_test, y_train, y_test = train_test_split(self.index, self.y, test_size=0.1, stratify=self.y)
        if train:
            self.index, self.y = X_train, y_train
        else:
            self.index, self.y = X_test, y_test
        self.classes = self.categories
        self.bc_transform = self.transform
        self.transform = lambda x: self.bc_transform(x.convert('RGB'))


class MyImageFolder(ImageFolder):
    def __init__(self, *args, train=True, **kwargs):
        super(MyImageFolder, self).__init__(*args, **kwargs)
        with open(kwargs['root'] + '../{}.txt'.format('train' if train else 'test')) as f:
            self.samples = [
                (os.path.join(kwargs['root'], img_path).strip('\n'), self.class_to_idx[img_path.split('/')[0]]) for
                img_path in f.readlines()]


class MyCUB200(MyImageFolder):
    def __init__(self, *args, download=False, **kwargs):
        kwargs['root'] += '/cub200/images/'
        if download:
            Path("data/zip").mkdir(exist_ok=True)
            download_cub200()
        super(MyCUB200, self).__init__(*args, **kwargs)


class MyMITIndoor(MyImageFolder):
    def __init__(self, *args, download=False, **kwargs):
        kwargs['root'] += '/mit_indoor/images/'
        if download:
            Path("data/zip").mkdir(exist_ok=True)
            download_mit_indoor()
        super(MyMITIndoor, self).__init__(*args, **kwargs)


class MyTinyImageNet200(ImageFolder):
    def __init__(self, *args, train=True, download=False, **kwargs):
        kwargs['root'] += f'/tiny_imagenet_200/{"train" if train else "val"}'
        if download:
            Path("data/zip").mkdir(exist_ok=True)
            download_tinyimagenet()
        super(MyTinyImageNet200, self).__init__(*args, **kwargs)