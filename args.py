import argparse


def get_args():
    parser = argparse.ArgumentParser(description='pytorch-image-classification', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    parser.add_argument('data_dir', type=str, help='dataset dir')
    parser.add_argument('-i', '--train-resize', type=int, default=(224, 224), nargs='+', help='train image size')
    parser.add_argument('-ti', '--test-resize', type=int, default=(224, 224), nargs='+', help='test image size')
    parser.add_argument('--crop-ptr', type=float, default=0.9, help='test image crop percent')
    parser.add_argument('--interpolation', type=str, default='bilinear', help='image interpolation mode')
    parser.add_argument('--mean', type=float, default=(0.5, 0.5, 0.5), nargs='+', help='image mean')
    parser.add_argument('--std', type=float, default=(0.5, 0.5, 0.5), nargs='+', help='image std')
    parser.add_argument('-hf', '--hflip', type=float, default=0.5, help='random horizontal flip')
    parser.add_argument('-aa', '--auto_aug', type=str, default=None, help='rand augmentation')
    parser.add_argument('-re', '--remode', type=float, default=0.2, help='random erasing probability')

    # model
    parser.add_argument('-m', '--model-name', type=str, default='resnet50', help='model name')

    # setup
    parser.add_argument('-proj', '--project-name', type=str, default='pytorch-image-classification', help='project name used for wandb logger')
    parser.add_argument('-exp', '--exp-name', type=str, default=None, help='experiment name for each run')
    parser.add_argument('-out', '--output-dir', type=str, default='log', help='where log output is saved')
    parser.add_argument('-s', '--seed', type=int, default=None, help='fix seed')
    parser.add_argument('--use-deterministic', action='store_true', default=False, help='use deterministic algorithm (slow, but better for reproduction)')

    return parser.parse_args()
