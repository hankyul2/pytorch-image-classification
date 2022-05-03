from torch.optim import SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, LambdaLR, SequentialLR, OneCycleLR, \
    MultiStepLR


def get_optimizer_and_scheduler(model, args):
    """get optimizer and scheduler
    :arg
        model: nn.Module instance
        args: argparse instance containing optimizer and scheduler hyperparameter
    """
    parameter = model.parameters()
    total_iter = args.epoch * args.iter_per_epoch
    warmup_iter = args.warmup_epoch * args.iter_per_epoch

    if args.optimizer == 'sgd':
        optimizer = SGD(parameter, args.lr, args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(parameter, args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(parameter, args.lr, eps=args.eps, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        NotImplementedError(f"{args.optimizer} is not supported yet")

    if args.scheduler == 'cosine':
        main_scheduler = CosineAnnealingLR(optimizer, total_iter-warmup_iter, args.min_lr)
    elif args.scheduler == 'multistep':
        main_scheduler = MultiStepLR(optimizer, args.milestones)
    elif args.scheduler == 'step':
        main_scheduler = StepLR(optimizer, total_iter-warmup_iter, gamma=args.decay_rate)
    elif args.scheduler =='explr':
        main_scheduler = ExponentialLR(optimizer, gamma=args.decay_rate)
    elif args.scheduler == 'onecyclelr':
        main_scheduler = OneCycleLR(optimizer, args.lr, total_iter, three_phase=args.three_phase)
    else:
        NotImplementedError(f"{args.scheduler} is not supported yet")

    if args.warmup_epoch and args.scheduler != 'onecyclelr':
        if args.warmup_scheduler == 'linear':
            lr_lambda = lambda e: (e * (args.lr - args.warmup_lr) / warmup_iter + args.warmup_lr) / args.lr
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            NotImplementedError(f"{args.warmup_scheduler} is not supported yet")

        scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], [warmup_iter])
    else:
        scheduler = main_scheduler

    return optimizer, scheduler