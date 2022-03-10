import os

import torch
from torch import distributed as dist


class Metric:
    def __init__(self, reduce_every_n_step=50, reduce_on_compute=True, header='', fmt='{val:.2f} ({avg:.2f})'):
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
        self.header = header
        self.fmt = fmt

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().clone()
        elif self.reduce_every_n_step and not isinstance(val, torch.Tensor):
            raise ValueError('reduce operation is allowed for only tensor')

        if self.val is None:
            self.n = n
            self.val = val
            self.sum = val
            self.avg = self.sum / self.n
        else:
            self.val = val
            self.sum += val * n
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
        return self.header + ' ' + self.fmt.format(**self.__dict__)


def Accuracy(y_hat, y, top_k=(1,)):
    """Compute top-k accuracy
    :arg
        y_hat(tensor): prediction shaped as (B, C)
        y(tensor): label shaped as (B)
        top_k(tuple): how exactly model should predict in each metric

    :return
        list of metric scores
    """
    prediction = torch.argsort(y_hat, dim=-1, descending=True)
    accuracy = [(prediction[:, :min(k, y_hat.size(1))] == y.unsqueeze(-1)).float().sum(dim=-1).mean() * 100 for k in top_k]
    return accuracy


def all_reduce_mean(val, world_size):
    """Collect value to each gpu
    :arg
        val(tensor): target
        world_size(int): the number of process in each group
    """
    dist.all_reduce(val.clone(), dist.ReduceOp.SUM)
    val = val / world_size
    return val


def reduce_mean(val, world_size):
    """Collect value to local zero gpu
    :arg
        val(tensor): target
        world_size(int): the number of process in each group
    """
    dist.reduce(val.clone(), 0, dist.ReduceOp.SUM)
    val = val / world_size
    return val