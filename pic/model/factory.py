from copy import copy

import timm
import torch
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel

from pic.model import ModelEmaV2, create_model


def get_model(args):
    if args.model_type == 'torchvision':
        model = torchvision.models.__dict__[args.model_name](num_classes=args.num_classes, pretrained=args.pretrained).cuda(args.device)
    elif args.model_type == 'timm':
        model = timm.create_model(args.model_name, in_chans=args.in_channels, num_classes=args.num_classes, drop_path_rate=args.drop_path_rate, pretrained=args.pretrained, checkpoint_path=args.pretrained_path).cuda(args.device)
    elif args.model_type == 'pic':
        kwargs = copy(args.__dict__)
        kwargs.pop('model_name')
        model = create_model(args.model_name, **kwargs).cuda(args.device)
    else:
        raise Exception(f"{args.model_type} is not supported yet")

    return model

def get_ema_ddp_model(model, args):
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.ema:
        ema_model = ModelEmaV2(model, args.ema_decay, None)
    else:
        ema_model = None

    if args.distributed:
        ddp_model = DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        ddp_model = None

    return model, ema_model, ddp_model

