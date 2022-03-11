import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='pytorch-image-classification', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    parser.add_argument('data_dir', type=str, help='dataset dir')
    parser.add_argument('-i', '--train-resize', type=int, default=(224, 224), nargs='+', help='train image size')
    parser.add_argument('-ti', '--test-resize', type=int, default=(224, 224), nargs='+', help='test image size')
    parser.add_argument('--crop-ptr', type=float, default=0.9, help='test image crop percent')
    parser.add_argument('--interpolation', type=str, default='bilinear', help='image interpolation mode')
    parser.add_argument('--mean', type=float, default=(0.5, 0.5, 0.5), nargs='+', help='image mean')
    parser.add_argument('--std', type=float, default=(0.5, 0.5, 0.5), nargs='+', help='image std')

    # augmentation
    parser.add_argument('-hf', '--hflip', type=float, default=0.5, help='random horizontal flip')
    parser.add_argument('-aa', '--auto-aug', type=str, default=None, help='rand augmentation')
    parser.add_argument('--cutmix', type=float, default=0.5, help='cutmix probability')
    parser.add_argument('--mixup', type=float, default=0.5, help='mix probability')
    parser.add_argument('-re', '--remode', type=float, default=0.2, help='random erasing probability')
    parser.add_argument('--aug-repeat', type=int, default=None, help='repeat augmentation')

    # model
    parser.add_argument('-m', '--model-name', type=str, default='resnet50', help='model name')
    parser.add_argument('--sync-bn', action='store_true', default=False, help='apply sync batchnorm')
    parser.add_argument('--ema', action='store_true', default=False, help='apply EMA for model')
    parser.add_argument('--ema-decay', type=float, default=0.99999, help='exponential model average decay')
    parser.add_argument('--pretrained', action='store_true', default=False, help='load pretrained weight')

    # criterion
    parser.add_argument('-c', '--criterion', type=str, default='ce', help='loss function')
    parser.add_argument('--smoothing', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--bce-target', type=float, default=None, help='remove cutmix/mixup label below target prob')

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate(lr)')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer name')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='optimizer weight decay')
    parser.add_argument('--nesterov', action='store_true', default=False, help='use nesterov momentum')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999], help='adam optimizer beta parameter')
    parser.add_argument('--eps', type=float, default=1e-6, help='optimizer eps')

    # gradient normalization
    parser.add_argument('--grad-norm', type=float, default=None, help='gradient clipping threshold')

    # scheduler
    parser.add_argument('--scheduler', type=str, default='cosine', help='lr scheduler')
    parser.add_argument('--step-size', type=int, default=2, help='lr decay step size')
    parser.add_argument('--decay-rate', type=float, default=0.1, help='lr decay rate')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='lowest lr used for cosine scheduler')
    parser.add_argument('--warmup-scheduler', type=str, default='linear', help='warmup lr scheduler type')
    parser.add_argument('--warmup-lr', type=float, default=1e-4, help='warmup start lr')
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')

    # train time
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('-j', '--num-workers', type=int, default=8, help='number of workers')
    parser.add_argument('--pin-memory', action='store_true', default=False, help='pin memory in dataloader')
    parser.add_argument('--amp', action='store_true', default=False, help='enable native amp(fp16) training')
    parser.add_argument('--channels-last', action='store_true', default=False, help='change memory format to channels last')
    parser.add_argument('--cuda', type=str, default='0,1,2,3,4,5,6,7,8', help='CUDA_VISIBLE_DEVICES options')

    # control logic (validate & resume & start & end epoch)
    parser.add_argument('--validate-only', action='store_true', default=False, help='if enabled, evaluate model')
    parser.add_argument('--resume', action='store_true', default=False, help='if true, resume from checkpoint_path')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='resume checkpoint path')
    parser.add_argument('--start-epoch', type=int, default=None, help='start of epoch(override resume epoch)')
    parser.add_argument('--end-epoch', type=int, default=None, help='early stop epoch')

    # setup
    parser.add_argument('--use-wandb', action='store_true', default=False, help='track std out and log metric in wandb')
    parser.add_argument('-proj', '--project-name', type=str, default='pytorch-image-classification', help='project name used for wandb logger')
    parser.add_argument('--who', type=str, default='Hankyul', help='enter your name')
    parser.add_argument('-exp', '--exp-name', type=str, default=None, help='experiment name for each run')
    parser.add_argument('-out', '--output-dir', type=str, default='log', help='where log output is saved')
    parser.add_argument('-p', '--print-freq', type=int, default=50, help='how often print metric in iter')
    parser.add_argument('-s', '--seed', type=int, default=None, help='fix seed')
    parser.add_argument('--use-deterministic', action='store_true', default=False, help='use deterministic algorithm (slow, but better for reproduction)')

    return parser
