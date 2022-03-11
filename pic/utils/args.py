import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='pytorch-image-classification', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    parser.add_argument('data_dir', type=str, help='dataset dir')

    # train input
    train_input = parser.add_argument_group('train_input')
    train_input.add_argument('--train-resize', type=int, default=(224, 224), nargs='+', help='train image size')
    train_input.add_argument('--interpolation', type=str, default='bilinear', help='image interpolation mode')
    train_input.add_argument('--mean', type=float, default=(0.485, 0.456, 0.406), nargs='+', help='image mean')
    train_input.add_argument('--std', type=float, default=(0.229, 0.224, 0.225), nargs='+', help='image std')

    # test input
    test_input = parser.add_argument_group('test_input')
    test_input.add_argument('--test-resize', type=int, default=(256, 256), nargs='+', help='test image size')
    test_input.add_argument('--test-resize-mode', type=str, default='resize_shorter', choices=['resize_shorter', 'resize'], help='test resize mode')
    test_input.add_argument('--crop-ptr', type=float, default=0.875, help='test image crop percent')

    # augmentation
    augmentation = parser.add_argument_group('argument')
    augmentation.add_argument('-hf', '--hflip', type=float, default=0.5, help='random horizontal flip')
    augmentation.add_argument('-aa', '--auto-aug', type=str, default=None, help='rand augmentation')
    augmentation.add_argument('--cutmix', type=float, default=0.5, help='cutmix probability')
    augmentation.add_argument('--mixup', type=float, default=0.5, help='mix probability')
    augmentation.add_argument('-re', '--remode', type=float, default=0.2, help='random erasing probability')
    augmentation.add_argument('--aug-repeat', type=int, default=None, help='repeat augmentation')

    # model
    model = parser.add_argument_group('model')
    model.add_argument('-m', '--model-name', type=str, default='resnet50', help='model name')
    model.add_argument('--sync-bn', action='store_true', default=False, help='apply sync batchnorm')
    model.add_argument('--ema', action='store_true', default=False, help='apply EMA for model')
    model.add_argument('--ema-decay', type=float, default=0.99999, help='exponential model average decay')
    model.add_argument('--pretrained', action='store_true', default=False, help='load pretrained weight')

    # criterion
    criterion = parser.add_argument_group('criterion')
    criterion.add_argument('-c', '--criterion', type=str, default='ce', help='loss function')
    criterion.add_argument('--smoothing', type=float, default=0.1, help='label smoothing')
    criterion.add_argument('--bce-target', type=float, default=None, help='remove cutmix/mixup label below target prob')

    # optimizer
    optimizer = parser.add_argument_group('optimizer')
    optimizer.add_argument('--lr', type=float, default=1e-3, help='learning rate(lr)')
    optimizer.add_argument('--epoch', type=int, default=100, help='epoch')
    optimizer.add_argument('--optimizer', type=str, default='adamw', help='optimizer name')
    optimizer.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    optimizer.add_argument('--weight-decay', type=float, default=1e-3, help='optimizer weight decay')
    optimizer.add_argument('--nesterov', action='store_true', default=False, help='use nesterov momentum')
    optimizer.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999], help='adam optimizer beta parameter')
    optimizer.add_argument('--eps', type=float, default=1e-6, help='optimizer eps')

    # gradient normalization
    grad_norm = parser.add_argument_group('grad_norm')
    grad_norm.add_argument('--grad-norm', type=float, default=None, help='gradient clipping threshold')

    # scheduler
    scheduler = parser.add_argument_group('scheduler')
    scheduler.add_argument('--scheduler', type=str, default='cosine', help='lr scheduler')
    scheduler.add_argument('--step-size', type=int, default=2, help='lr decay step size')
    scheduler.add_argument('--decay-rate', type=float, default=0.1, help='lr decay rate')
    scheduler.add_argument('--min-lr', type=float, default=1e-6, help='lowest lr used for cosine scheduler')
    scheduler.add_argument('--warmup-scheduler', type=str, default='linear', help='warmup lr scheduler type')
    scheduler.add_argument('--warmup-lr', type=float, default=1e-4, help='warmup start lr')
    scheduler.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')

    # train time
    train_time = parser.add_argument_group('train_time')
    train_time.add_argument('-b', '--batch-size', type=int, default=256, help='batch size')
    train_time.add_argument('-j', '--num-workers', type=int, default=8, help='number of workers')
    train_time.add_argument('--pin-memory', action='store_true', default=False, help='pin memory in dataloader')
    train_time.add_argument('--amp', action='store_true', default=False, help='enable native amp(fp16) training')
    train_time.add_argument('--channels-last', action='store_true', default=False, help='change memory format to channels last')
    train_time.add_argument('--cuda', type=str, default='0,1,2,3,4,5,6,7,8', help='CUDA_VISIBLE_DEVICES options')

    # control logic (validate & resume & start & end epoch)
    control = parser.add_argument_group('control')
    control.add_argument('--validate-only', action='store_true', default=False, help='if enabled, evaluate model')
    control.add_argument('--resume', action='store_true', default=False, help='if true, resume from checkpoint_path')
    control.add_argument('--checkpoint-path', type=str, default=None, help='resume checkpoint path')
    control.add_argument('--start-epoch', type=int, default=None, help='start of epoch(override resume epoch)')
    control.add_argument('--end-epoch', type=int, default=None, help='early stop epoch')

    # setup
    setup = parser.add_argument_group('setup')
    setup.add_argument('--use-wandb', action='store_true', default=False, help='track std out and log metric in wandb')
    setup.add_argument('-proj', '--project-name', type=str, default='pytorch-image-classification', help='project name used for wandb logger')
    setup.add_argument('--who', type=str, default='Hankyul', help='enter your name')
    setup.add_argument('-exp', '--exp-name', type=str, default=None, help='experiment name for each run')
    setup.add_argument('--exp-target', type=str, default=['model_name'], help='experiment target')
    setup.add_argument('-out', '--output-dir', type=str, default='log', help='where log output is saved')
    setup.add_argument('-p', '--print-freq', type=int, default=50, help='how often print metric in iter')
    setup.add_argument('-s', '--seed', type=int, default=None, help='fix seed')
    setup.add_argument('--use-deterministic', action='store_true', default=False, help='use deterministic algorithm (slow, but better for reproduction)')

    return parser
