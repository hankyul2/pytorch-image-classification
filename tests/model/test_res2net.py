import torch

from pic.model import create_model


def test_res2net34_18w_4s():
    x = torch.rand([2, 3, 224, 224])
    model = create_model('res2net34_18w_4s')
    y = model(x)
    assert list(y.shape) == [2, 1000]


def test_res2net50_18w_4s():
    x = torch.rand([2, 3, 224, 224])
    model = create_model('res2net50_18w_4s')
    y = model(x)
    assert list(y.shape) == [2, 1000]


def test_seres2net():
    x = torch.rand([2, 3, 224, 224])
    model = create_model('seres2net50_26w_4s')
    y = model(x)
    assert list(y.shape) == [2, 1000]


def test_seres2next():
    x = torch.rand([2, 3, 224, 224])
    model = create_model('seres2net50_26w_4s')
    y = model(x)
    assert list(y.shape) == [2, 1000]
