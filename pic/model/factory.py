import torch
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel

from pic.model import ModelEmaV2


def get_model(args):
    model = torchvision.models.__dict__[args.model_name](num_classes=args.num_classes, pretrained=args.pretrained).cuda(args.device)

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
