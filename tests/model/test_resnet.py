import torch

from pic.model import create_model


def test_senet():
    x = torch.rand([2, 3, 224, 224])
    for model_name in ['seresnext50_32_4', 'seresnet50', 'seresnet34']:
        model = create_model(model_name)
        y = model(x)
        assert list(y.shape) == [2, 1000]