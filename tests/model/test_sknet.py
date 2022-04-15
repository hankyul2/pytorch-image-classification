import torch
from torch import nn

from pic.model import create_model
from pic.model.sknet import SelectiveKernel


def test_selective_kernel_forward():
    x = torch.rand([2, 512, 28, 28])
    sk_conv = SelectiveKernel(512, 32, 2, nn.BatchNorm2d)
    y = sk_conv(x)
    assert list(y.shape) == [2, 512, 14, 14]


def test_sknet_forward():
    x = torch.rand([2, 3, 224, 224])
    model = create_model('sknet26_32_4')
    y = model(x)
    assert list(y.shape) == [2, 1000]

def test_sknet_original_forward():
    x = torch.rand([2, 3, 224, 224])
    model = create_model('sknet26')
    y = model(x)
    assert list(y.shape) == [2, 1000]

def test_sknet_timm_forward():
    x = torch.rand([2, 3, 224, 224])
    model = create_model('sknet26_timm')
    y = model(x)
    assert list(y.shape) == [2, 1000]

def test_sknet_ensemble_forward():
    x = torch.rand([2, 3, 224, 224])
    model = create_model('sknet26_ensemble')
    y = model(x)
    assert list(y.shape) == [2, 1000]
