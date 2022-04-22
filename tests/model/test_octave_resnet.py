import torch

from pic.model import create_model
from pic.model.octave_resnet import OctaveConvNormAct


def test_octave_conv_forward_stage_12():
    x = torch.rand([2, 64, 56, 56])
    conv1_1 = OctaveConvNormAct(64, 64, 0, 0.5)
    y = conv1_1(x)
    assert len(y) == 2
    assert list(y[0].shape) == [2, 32, 56, 56]
    assert list(y[1].shape) == [2, 32, 28, 28]

    conv1_2 = OctaveConvNormAct(64, 64, 0.5, 0.5, 3, 1, 1)
    y = conv1_2(y)
    assert len(y) == 2
    assert list(y[0].shape) == [2, 32, 56, 56]
    assert list(y[1].shape) == [2, 32, 28, 28]

    conv1_3 = OctaveConvNormAct(64, 256, 0.5, 0.5)
    y = conv1_3(y)
    assert len(y) == 2
    assert list(y[0].shape) == [2, 128, 56, 56]
    assert list(y[1].shape) == [2, 128, 28, 28]

    conv2_1 = OctaveConvNormAct(256, 128, 0.5, 0.5)
    y = conv2_1(y)
    assert len(y) == 2
    assert list(y[0].shape) == [2, 64, 56, 56]
    assert list(y[1].shape) == [2, 64, 28, 28]

    conv2_2 = OctaveConvNormAct(128, 128, 0.5, 0.5, 3, 2, 1)
    y = conv2_2(y)
    assert len(y) == 2
    assert list(y[0].shape) == [2, 64, 28, 28]
    assert list(y[1].shape) == [2, 64, 14, 14]

def test_octave_conv_forward_stage_4():
    x = (torch.rand([2, 512, 14, 14]), torch.rand([2, 512, 7, 7]))
    conv4_1 = OctaveConvNormAct(1024, 512, 0.5, 0)
    y = conv4_1(x)
    assert len(y) == 2
    assert list(y[0].shape) == [2, 512, 14, 14]
    assert y[1] is None

    conv4_2 = OctaveConvNormAct(512, 512, 0, 0, 3, 2, 1)
    y = conv4_2(y)
    assert len(y) == 2
    assert list(y[0].shape) == [2, 512, 7, 7]
    assert y[1] is None

def test_oct_resnet50_alpha5():
    x = torch.rand([2, 3, 224, 224])
    f = create_model('oct_resnet50_alpha5')
    y = f(x)
    assert list(y.shape) == [2, 1000]