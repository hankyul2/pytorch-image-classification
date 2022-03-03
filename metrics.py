import os

import torch
from torch import distributed as dist


class BaseMetric:
    def __init__(self, reduce_every_n_step=50, reduce_on_compute=False, fmt='{val:.2f} ({avg:.2f})'):
        """Base Metric Class supporting ddp setup

        :arg
            reduce_ever_n_step(int): call all_reduce every n step in ddp mode
            reduce_on_compute(bool): call all_reduce in compute() method
            fmt(str): format representing metric in string
        """
        self.dist = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

        if self.dist:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.reduce_every_n_step = reduce_every_n_step
            self.reduce_on_compute = reduce_on_compute
        else:
            self.world_size = None
            self.reduce_every_n_step = self.reduce_on_compute = False

        self.val = None
        self.sum = None
        self.avg = None
        self.n = None
        self.fmt = fmt

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach()
        elif self.reduce_every_n_step and not isinstance(val, torch.Tensor):
            raise ValueError('reduce operation is allowed for only tensor')

        if self.val is None:
            self.n = n
            self.val = val
            self.sum = val
            self.avg = self.sum / self.n
        else:
            self.val = val
            self.sum += val
            self.n += n
            self.avg = self.sum / self.n

        if self.reduce_every_n_step and n % self.reduce_every_n_step == 0:
            self.sum = all_reduce_mean(self.sum, self.world_size)
            self.avg = self.sum / self.n

    def compute(self):
        if self.reduce_on_compute:
            self.sum = all_reduce_mean(self.sum, self.world_size)
            self.avg = self.sum / self.n

        return self.avg

    def reset(self):
        self.val = None
        self.n = None
        self.sum = None
        self.avg = None

    def __str__(self):
        return self.fmt.format(**self.__dict__)


def Accuracy(y_hat, y, top_k=(1,)):
    prediction = torch.argsort(y_hat, dim=-1, descending=True)
    accuracy = [(prediction[:, :k] == y.unsqueeze(-1)).float().sum(dim=-1).mean() for k in top_k]
    return accuracy


def all_reduce_mean(val, world_size):
    dist.all_reduce(val.detach(), dist.ReduceOp.SUM)
    val = val / world_size
    return val


def reduce_mean(val, world_size):
    dist.reduce(val.detach(), 0, dist.ReduceOp.SUM)
    val = val / world_size
    return val