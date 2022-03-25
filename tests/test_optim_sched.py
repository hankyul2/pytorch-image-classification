import copy

import pytest
import torch
from box import Box
from torch import nn

from pic.criterion.scaler import NativeScalerWithGradAccum
from pic.criterion.bce import BinaryCrossEntropy
from pic.optimizer.factory import get_optimizer_and_scheduler


@pytest.fixture()
def args():
    return Box({'optimizer':'sgd', 'epoch': 10, 'lr': 1e-3,
                'momentum': 0.9, 'weight_decay': 1e-3, 'nesterov': False,
                'betas': [0.9, 0.999], 'eps': 1e-6,
                'scheduler':'cosine', 'step_size':2, 'decay_rate':0.6, 'min_lr': 1e-6,
                'warmup_scheduler': 'linear', 'warmup_lr': 1e-4, 'warmup_epoch': 5})

@pytest.fixture()
def model():
    return nn.Linear(5, 5)

def test_get_sgd_linear(model, args):
    args.optimizer = 'sgd'
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    lr = []
    computed_lr = [args.warmup_lr, 0.00028, 0.00046, 0.00064, 0.00082, 0.001]
    for epoch in range(args.epoch):
        lr.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    assert sum([abs(l - c) for l, c in zip(lr[:6], computed_lr)]) < 1e-6

def test_get_adamw_linear(model, args):
    args.optimizer = 'adamw'
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    lr = []
    computed_lr = [args.warmup_lr, 0.00028, 0.00046, 0.00064, 0.00082, 0.001]
    for epoch in range(args.epoch):
        lr.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    assert sum([abs(l - c) for l, c in zip(lr[:6], computed_lr)]) < 1e-6

def test_get_rmsprop_linear(model, args):
    args.optimizer = 'rmsprop'
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    lr = []
    computed_lr = [args.warmup_lr, 0.00028, 0.00046, 0.00064, 0.00082, 0.001]
    for epoch in range(args.epoch):
        lr.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    assert sum([abs(l - c) for l, c in zip(lr[:6], computed_lr)]) < 1e-6

def test_get_sgd_step(model, args):
    args.optimizer = 'sgd'
    args.scheduler = 'step'
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    lr = []
    computed_lr = [args.warmup_lr, 0.00028, 0.00046, 0.00064, 0.00082, 0.001]
    for epoch in range(args.epoch):
        lr.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    assert sum([abs(l - c) for l, c in zip(lr[:6], computed_lr)]) < 1e-6

def test_get_sgd_exp(model, args):
    args.optimizer = 'sgd'
    args.scheduler = 'explr'
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    lr = []
    computed_lr = [args.warmup_lr, 0.00028, 0.00046, 0.00064, 0.00082, 0.001]
    for epoch in range(args.epoch):
        lr.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

def test_bce_label_smoothing():
    bce_loss_fn = BinaryCrossEntropy(label_smoothing=0.0)
    bce_label_smoothing_loss_fn = BinaryCrossEntropy(label_smoothing=0.1)

    logits = torch.rand([5, 10])
    target = torch.randint(10, (5,))

    bce_loss = bce_loss_fn(logits, target)

def test_scaler(model):
    model = model.cuda()
    model2 = copy.deepcopy(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)

    criterion = BinaryCrossEntropy(label_smoothing=0.1)
    scaler = NativeScalerWithGradAccum()

    x = torch.rand([5, 5]).cuda()
    y = torch.randint(5, (5,)).cuda()

    with torch.autocast('cuda', True):
        pred = model(x)
        pred2 = model2(x)

        loss = criterion(pred, y)
        loss2 = criterion(pred2, y)

    scaler(loss, optimizer, model.parameters())
    scaler(loss2, optimizer2, model2.parameters(), grad_norm=1.0)

    if (model2.weight.grad < 1.0).float().mean() > 0.5:
        assert (model.weight - model2.weight).abs_().sum() < 1e-6
    else:
        assert (model.weight - model2.weight).abs_().sum() > 1e-6







