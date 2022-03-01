import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR, SequentialLR, ExponentialLR


def get_optimizer_and_scheduler(model, args):
    """get optimizer and scheduler
    :arg
        model: nn.Module instance
        args: argparse instance containing optimizer and scheduler hyperparameter
    """
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


class BinaryCrossEntropy(nn.Module):
    def __init__(self, label_smoothing=0.1, bce_target=None):
        """Binary Cross Entropy (timm)
        :arg
            label_smoothing: multi-class loss in bce.
            bce_target: remove uncertain target used with cutmix.
        """
        super(BinaryCrossEntropy, self).__init__()
        self.smoothing = label_smoothing
        self.bce_target = bce_target

    def forward(self, x, y):
        if x.shape != y.shape:
            smooth = self.smoothing / x.size(-1)
            label = 1.0 - self.smoothing + smooth
            smooth_target = torch.full_like(x, smooth)
            y = torch.scatter(smooth_target, -1, y.long().view(-1, 1), label)
        if self.bce_target:
            y = torch.gt(y, self.bce_target).long()
        return F.binary_cross_entropy_with_logits(x, y, reduction='mean')


class NativeScalerWithGradAccum:
    def __init__(self):
        """NativeScalerWithGradAccum (timm)
        Native(pytorch) f16 scaler
        """
        self._scaler = GradScaler()

    def __call__(self, loss, optimizer, model_param, grad_norm=None, update=True):
        self._scaler.scale(loss).backward()
        if update:
            if grad_norm:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_param, grad_norm)
            self._scaler.step(optimizer)
            self._scaler.update()


def get_scaler_criterion(args):
    """Get Criterion(Loss) function and scaler
    Criterion functions are divided depending on usage of mixup
    - w/ mixup - you don't need to add smoothing loss, because mixup will add smoothing loss.
    - w/o mixup - you should need to add smoothing loss
    """
    if args.use_mixup:
        if args.criterion == 'ce' or args.criterion == 'crossentropy':
            criterion = nn.CrossEntropyLoss()
        elif args.criterion == 'bce' or args.criterion == 'binarycrossentropy':
            criterion = BinaryCrossEntropy(bce_target=args.bec_target)
    else:
        if args.criterion == 'ce' or args.criterion == 'crossentropy':
            criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
        elif args.criterion == 'bce' or args.criterion == 'binarycrossentropy':
            criterion = BinaryCrossEntropy(label_smoothing=args.smoothing, bce_target=args.bec_target)

    valid_criterion = nn.CrossEntropyLoss()

    if args.amp:
        scaler = NativeScalerWithGradAccum()
    else:
        scaler = None

    return criterion, valid_criterion, scaler

