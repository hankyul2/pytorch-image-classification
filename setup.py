import os
import random
from datetime import datetime
from pathlib import Path

import numpy
import torch


def setup_for_distributed(is_master):
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

        setup_for_distributed(args.gpu == 0)
    else:
        print('| Not using distributed mode')
        args.distributed = False
        args.gpu = 0
    return


def setup(args):
    init_distributed_mode(args)

    if args.exp_name is None:
        args.exp_name = f'{args.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        args.log_dir = os.path.join(args.output_dir, args.exp_name)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    args.device = torch.device(f'cuda:{args.gpu}')

    print("| Project Name:", args.project_name)
    print("| Experiment Name:", args.exp_name)
    print("| Model Name:", args.model_name)

    if args.seed is not None:
        numpy.random.seed(args.seed)
        random.seed(args.seed)
        torch.random(args.seed)

    if args.use_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True