from torch.optim import SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR, SequentialLR, ExponentialLR


def get_optimizer_and_scheduler(model, args):
    parameter = model.parameters()

    if args.optimizer == 'sgd':
        optimizer = SGD(parameter, args.lr, args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(parameter, args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(parameter, args.lr, eps=args.eps, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        NotImplementedError(f"{args.optimizer} is not supported yet")

    if args.scheduler == 'cosine':
        main_scheduler = CosineAnnealingLR(optimizer, args.epoch, args.min_lr)
    elif args.scheduler == 'step':
        main_scheduler = StepLR(optimizer, args.step_size, gamma=args.decay_rate)
    elif args.scheduler =='explr':
        main_scheduler = ExponentialLR(optimizer, gamma=args.decay_rate)
    else:
        NotImplementedError(f"{args.scheduler} is not supported yet")

    if args.warmup_epoch:
        if args.warmup_scheduler == 'linear':
            lr_lambda = lambda e: (e * (args.lr - args.warmup_lr) / args.warmup_epoch + args.warmup_lr) / args.lr
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            NotImplementedError(f"{args.warmup_scheduler} is not supported yet")

        scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], [args.warmup_epoch])
    else:
        scheduler = main_scheduler

    return optimizer, scheduler