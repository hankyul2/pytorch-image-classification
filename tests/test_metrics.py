import torch

from pic.utils.metrics import Metric, Accuracy


def test_init():
    metric = Metric()
    assert metric is not None


def test_update():
    metric = Metric()
    metric.update(0.5)
    assert metric.val == 0.5
    assert metric.n == 1
    metric.update(0.5)
    assert metric.sum == 1.0
    assert metric.n == 2

def test_avg():
    metric = Metric()
    metric.update(0.5)
    metric.update(0.5)
    assert metric.avg == 0.5

def test_reset():
    metric = Metric()
    metric.update(0.5)
    metric.update(0.5)
    metric.reset()
    assert metric.val is None

def test_compute():
    metric = Metric()
    metric.update(0.1)
    metric.update(0.5)
    metric.update(0.9)
    assert metric.compute() == 0.5

def test_str_format():
    metric = Metric()
    metric.update(0.1)
    metric.update(0.7)
    metric.update(0.8)
    metric.update(-0.8)
    metric.update(-0.7)
    assert str(metric) == ' -0.70 (0.02)'

def test_top1_accuracy_update():
    y_hat = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    y = torch.tensor([1, 1])
    top1 = Accuracy(y_hat, y, top_k=(1,))[0]
    assert top1.item() == 50.0

def test_top5_accuracy_update():
    y_hat = torch.tensor([[0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1], [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]])
    y = torch.tensor([2, 6])
    top5 = Accuracy(y_hat, y, top_k=(5,))[0]
    assert top5.item() == 50.0

