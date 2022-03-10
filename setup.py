import glob
from datetime import datetime
import logging
import os
import random
from functools import partial
from pathlib import Path

import numpy
import torch
import wandb


def allow_print_to_master(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)

        if force or is_master:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    print(f'{datetime.now().strftime("[%Y/%m/%d %Hh %Mm %Ss]")} ', end='')

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_backend = 'nccl'
        args.dist_url = 'env://'

        print(f'| distributed init (rank {args.rank}): {args.dist_url}')
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
    else:
        print('| Not using distributed mode')
        args.distributed = False
        args.gpu = 0

    args.is_rank_zero = args.gpu == 0
    allow_print_to_master(args.is_rank_zero)
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(f'cuda:{args.gpu}')


def make_logger(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", "[%Y/%m/%d %Hh %Mm %Ss]")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log(msg, metric=False, logger=None):
    if logger:
        if metric:
            wandb.log(msg)
        else:
            logger.info(msg)


def init_logger(args):
    args.version_id = len(list(glob.glob(os.path.join(args.output_dir, f'{args.exp_name}_v*'))))
    args.exp_name = f'{args.exp_name}_v{args.version_id}'
    args.log_dir = os.path.join(args.output_dir, args.exp_name)
    args.text_log_path = os.path.join(args.log_dir, 'log.txt')
    args.best_weight_path = os.path.join(args.log_dir, 'best_weight.pth')

    if args.is_rank_zero:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        args.logger = make_logger(args.text_log_path)
        if args.use_wandb:
            wandb.init(project=args.project_name, name=args.exp_name, config=args)
    else:
        args.logger = None

    args.log = partial(log, logger=args.logger)

    args.log("| Project Name: %s" % args.project_name)
    args.log("| Experiment Name: %s" % args.exp_name)
    args.log('| Experiment Data: %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    args.log("| Model Name: %s" % args.model_name)
    args.log("| Log dir: %s" % args.log_dir)


def setup(args):
    init_distributed_mode(args)
    init_logger(args)

    if args.seed is not None:
        numpy.random.seed(args.seed)
        random.seed(args.seed)
        torch.random(args.seed)

    if args.use_deterministic:
        # Todo: considering add args.validate_only as condition
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True