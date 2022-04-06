from torch import nn

from pic.criterion import NativeScalerWithGradAccum, BinaryCrossEntropy


def get_scaler_criterion(args):
    """Get Criterion(Loss) function and scaler
    Criterion functions are divided depending on usage of mixup
    - w/ mixup - you don't need to add smoothing loss, because mixup will add smoothing loss.
    - w/o mixup - you should need to add smoothing loss
    """
    if args.criterion in ['ce', 'crossentropy']:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    elif args.criterion in ['bce', 'binarycrossentropy']:
        criterion = BinaryCrossEntropy(label_smoothing=args.smoothing, bce_target=args.bec_target)

    valid_criterion = nn.CrossEntropyLoss()

    if args.amp:
        scaler = NativeScalerWithGradAccum()
    else:
        scaler = None

    return criterion, valid_criterion, scaler