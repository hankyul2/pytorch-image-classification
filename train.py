# How to run this?
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 pt_elastic.py

# What does each variable mean?
# CUDA_DEVICE_ORDER - control device order
# CUDA_VISIBLE_DEVICES - control device id
# nproc_per_node - control parallelism
# device - control between cpu and gpu

# Other required procedures
# First, distributed sampler or distributed sampler based sampler such as random sampler
# Second, `torch.nn.parallel.DistributedParallel(model, deivce_ids=[args.gpu])`
# Third, `set_epoch(epoch)` for sampler

import os
import argparse
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
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_backend = 'nccl'
        args.dist_url = 'env://'

        torch.cuda.set_device(args.gpu)
        print(f'| distributed init (rank {args.rank}): {args.dist_url}')
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)

        setup_for_distributed(args.gpu == 0)
    else:
        print('Not using distributed mode')
        args.distributed = False
    return


def main(args):
    init_distributed_mode(args)

    print("| Project Name:", args.description)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='pytorch-image-classification', add_help=True)

    main(args)