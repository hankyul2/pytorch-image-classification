import pytest
import torch
from PIL import Image

from pic.data.transforms import TrainTransform, ValTransform
from pic.data.mix import MixUP, CutMix, mix_target, get_cutmix_bounding_box


@pytest.fixture
def iu():
    img = Image.open('iu.jpg')
    yield img
    img.close()

@pytest.fixture
def twice():
    img = Image.open('twice.jpg')
    yield img
    img.close()

def test_image_net_train(iu):
    transform_fn = TrainTransform((224, 224), 0.5, 'ra', 0.2, 'bilinear', (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    x = transform_fn(iu)
    x_numpy = x.detach().permute(1, 2, 0).numpy()
    assert list(x.shape) == [3, 224, 224]

def test_image_net_val(twice):
    img = Image.new(mode='RGB', size=(256, 256))
    transform_fn = ValTransform((224, 224), 0.95, 'bilinear', (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    x = transform_fn(img)
    assert list(x.shape) == [3, 224, 224]

def test_mixup_target():
    nclass = 10
    y1 = torch.randint(0, 10, (nclass,))
    y2 = torch.randint(0, 10, (nclass,))

    y = mix_target(y1, y2, 0.5, nclass=nclass)

    assert list(y.shape) == [10, nclass]

def test_get_cutmix_boudning_box():
    img_shape = [32, 32]
    ratio = 0.5
    (x_s, x_e, y_s, y_e), ratio = get_cutmix_bounding_box(img_shape, ratio)
    assert 0 <= x_s <= 32
    assert 0 <= x_e <= 32
    assert 0 <= y_s <= 32
    assert 0 <= y_e <= 32

def test_mixup(iu, twice):
    transform_fn = TrainTransform((224, 224), 0.5, 'ra', 0.2, 'bilinear', (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    x1 = transform_fn(iu)
    x2 = transform_fn(twice)
    assert list(x1.shape) == [3, 224, 224]
    assert list(x2.shape) == [3, 224, 224]

    x = torch.stack([x1, x2])
    y = torch.tensor([0, 1])

    collate_fn = MixUP()
    new_x, new_y = collate_fn(x, y)

    assert list(new_x.shape) == [2, 3, 224, 224]


def test_cutmix(iu, twice):
    transform_fn = TrainTransform((224, 224), 0.5, 'ra', 0.2, 'bilinear', (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    x1 = transform_fn(iu)
    x2 = transform_fn(twice)
    assert list(x1.shape) == [3, 224, 224]
    assert list(x2.shape) == [3, 224, 224]

    x = torch.stack([x1, x2])
    y = torch.tensor([0, 1])

    collate_fn = CutMix()
    new_x, new_y = collate_fn(x, y)

    assert list(new_x.shape) == [2, 3, 224, 224]
