import os
import zipfile
import gdown
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision.datasets import Caltech101, ImageFolder, StanfordCars, Flowers102

from pic.data.dataset import register_dataset


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


@register_dataset
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


@register_dataset
class MyStanfordCars(StanfordCars):
    def __init__(self, *args, train=True, **kwargs):
        super(MyStanfordCars, self).__init__(*args, split='train' if train else 'test', **kwargs)


@register_dataset
class MyFlowers102(Flowers102):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.split = ('train', 'val') if train else ('test',)
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate(labels["labels"].tolist()))
        self._labels = []
        self._image_files = []
        self.classes = [i for i in range(102)]

        for split in self.split:
            for image_id in set_ids[self._splits_map[split]].tolist():
                self._labels.append(image_id_to_label[image_id-1]-1)
                self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

@register_dataset
class MyImageFolder(ImageFolder):
    def __init__(self, *args, train=True, download=False, **kwargs):
        super(MyImageFolder, self).__init__(*args, **kwargs)
        with open(kwargs['root'] + '../{}.txt'.format('train' if train else 'test')) as f:
            self.samples = [
                (os.path.join(kwargs['root'], img_path).strip('\n'), self.class_to_idx[img_path.split('/')[0]]) for
                img_path in f.readlines()]


@register_dataset
class CUB200(MyImageFolder):
    def __init__(self, *args, download=False, **kwargs):
        kwargs['root'] += '/cub200/images/'
        if download:
            Path("data/zip").mkdir(exist_ok=True)
            download_cub200()
        super(CUB200, self).__init__(*args, **kwargs)


@register_dataset
class MiTIndoor(MyImageFolder):
    def __init__(self, *args, download=False, **kwargs):
        kwargs['root'] += '/mit_indoor/images/'
        if download:
            Path("data/zip").mkdir(exist_ok=True)
            download_mit_indoor()
        super(MiTIndoor, self).__init__(*args, **kwargs)


@register_dataset
class TinyImageNet(ImageFolder):
    def __init__(self, *args, train=True, download=False, **kwargs):
        kwargs['root'] += f'/tiny_imagenet_200/{"train" if train else "val"}'
        if download:
            Path("data/zip").mkdir(exist_ok=True)
            download_tinyimagenet()
        super(TinyImageNet, self).__init__(*args, **kwargs)
