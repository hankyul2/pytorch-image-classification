import os
import sys
import argparse

from pathlib import Path
from traceback import print_exc

from train import run
from pic.utils import get_args_parser, clear


setting_dict = dict(
    # best settings @ 32
    cifar10_32="data --dataset_type CIFAR10 --train-size 32 32 --train-resize-mode ResizeRandomCrop --random-crop-pad 4 --test-size 32 32 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4802 0.4481 0.3975 --std 0.2302 0.2265 0.2262 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.0 --smoothing 0.0 --epoch 300 --optimizer sgd --nesterov --lr 1 --min-lr 1e-4 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine -b 256 -j 4 --pin-memory --amp --channels-last",
    cifar100_32="data --dataset_type CIFAR100 --train-size 32 32 --train-resize-mode ResizeRandomCrop --random-crop-pad 4 --test-size 32 32 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4914 0.4825 0.4467 --std 0.2471 0.2435 0.2616 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.0 --smoothing 0.0 --epoch 300 --optimizer sgd --nesterov --lr 1 --min-lr 1e-4 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine -b 256 -j 4 --pin-memory --amp --channels-last",
    fashion_32="data --dataset_type FashionMNIST --in-channels 1 --train-size 32 32 --train-resize-mode ResizeRandomCrop --random-crop-pad 4 --test-size 32 32 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.2861 --std 0.3205 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.0 --smoothing 0.0 --epoch 300 --optimizer sgd --nesterov --lr 1 --min-lr 1e-4 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine -b 256 -j 4 --pin-memory --amp --channels-last",

    # best settings @ 224
    cifar10_224="data --dataset_type CIFAR10 --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4802 0.4481 0.3975 --std 0.2302 0.2265 0.2262 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.0 --smoothing 0.0 --epoch 300 --optimizer sgd --nesterov --lr 1 --min-lr 1e-4 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine -b 256 -j 4 --pin-memory --amp --channels-last",
    cifar100_224="data --dataset_type CIFAR100 --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4914 0.4825 0.4467 --std 0.2471 0.2435 0.2616 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.0 --smoothing 0.0 --epoch 300 --optimizer sgd --nesterov --lr 1 --min-lr 1e-4 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine -b 256 -j 4 --pin-memory --amp --channels-last",
    fashion_224="data --dataset_type FashionMNIST --in-channels 1 --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.2861 --std 0.3205 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.0 --smoothing 0.0 --epoch 300 --optimizer sgd --nesterov --lr 1 --min-lr 1e-4 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine -b 256 -j 4 --pin-memory --amp --channels-last",

    # hyperparameter search
    tiny_a_25="data --dataset_type TinyImageNet --train-size 128 128 --train-resize-mode ResizeRandomCrop --random-crop-pad 16 --test-size 128 128 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4802 0.4481 0.3975 --std 0.2302 0.2265 0.2262 --cutmix 0.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.1 --epoch 25 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    tiny_a_50 = "data --dataset_type TinyImageNet --train-size 128 128 --train-resize-mode ResizeRandomCrop --random-crop-pad 16 --test-size 128 128 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4802 0.4481 0.3975 --std 0.2302 0.2265 0.2262 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.1 --epoch 50 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    tiny_a_100 = "data --dataset_type TinyImageNet --train-size 128 128 --train-resize-mode ResizeRandomCrop --random-crop-pad 16 --test-size 128 128 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4802 0.4481 0.3975 --std 0.2302 0.2265 0.2262 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.1 --epoch 100 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    tiny_b_100 = "data --dataset_type TinyImageNet --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4802 0.4481 0.3975 --std 0.2302 0.2265 0.2262 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.0 --epoch 100 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    tiny_b_150 = "data --dataset_type TinyImageNet --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4802 0.4481 0.3975 --std 0.2302 0.2265 0.2262 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.0 --epoch 150 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    tiny_b_200 = "data --dataset_type TinyImageNet --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4802 0.4481 0.3975 --std 0.2302 0.2265 0.2262 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.0 --epoch 200 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    cifar100_a_25 = "data --dataset_type CIFAR100 --train-size 128 128 --train-resize-mode ResizeRandomCrop --random-crop-pad 16 --test-size 128 128 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.5070 0.4865 0.4409 --std 0.2009 0.1984 0.2023 --cutmix 0.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.1 --epoch 25 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    cifar100_a_50 = "data --dataset_type CIFAR100 --train-size 128 128 --train-resize-mode ResizeRandomCrop --random-crop-pad 16 --test-size 128 128 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.5070 0.4865 0.4409 --std 0.2009 0.1984 0.2023 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.1 --epoch 50 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    cifar100_a_100 = "data --dataset_type CIFAR100 --train-size 128 128 --train-resize-mode ResizeRandomCrop --random-crop-pad 16 --test-size 128 128 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.5070 0.4865 0.4409 --std 0.2009 0.1984 0.2023 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.1 --epoch 100 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    cifar100_b_100 = "data --dataset_type CIFAR100 --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.5070 0.4865 0.4409 --std 0.2009 0.1984 0.2023 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.0 --epoch 100 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    cifar100_b_150 = "data --dataset_type CIFAR100 --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.5070 0.4865 0.4409 --std 0.2009 0.1984 0.2023 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.0 --epoch 150 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    cifar100_b_200 = "data --dataset_type CIFAR100 --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.5070 0.4865 0.4409 --std 0.2009 0.1984 0.2023 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.0 --epoch 200 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    cifar10_a_25 = "data --dataset_type CIFAR10 --train-size 128 128 --train-resize-mode ResizeRandomCrop --random-crop-pad 16 --test-size 128 128 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --cutmix 0.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.1 --epoch 25 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 4 --pin-memory --amp --channels-last",
    cifar10_a_50 = "data --dataset_type CIFAR10 --train-size 128 128 --train-resize-mode ResizeRandomCrop --random-crop-pad 16 --test-size 128 128 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.1 --epoch 50 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 4 --pin-memory --amp --channels-last",
    cifar10_a_100 = "data --dataset_type CIFAR10 --train-size 128 128 --train-resize-mode ResizeRandomCrop --random-crop-pad 16 --test-size 128 128 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.1 --epoch 100 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 4 --pin-memory --amp --channels-last",
    cifar10_b_100 = "data --dataset_type CIFAR10 --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.0 --epoch 100 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    cifar10_b_150 = "data --dataset_type CIFAR10 --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.0 --epoch 150 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    cifar10_b_200 = "data --dataset_type CIFAR10 --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.0 --epoch 200 --lr 1e-2 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last",
    fashion_a_50="data --dataset_type FashionMNIST --in-channels 1 --train-size 128 128 --train-resize-mode ResizeRandomCrop --random-crop-pad 16 --test-size 128 128 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.2861 --std 0.3205 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.1 --smoothing 0.1 --epoch 50 --lr 1e-3 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 8 --pin-memory --amp --channels-last --print-flops",
)


def get_multi_args_parser():
    parser = argparse.ArgumentParser(description='pytorch-image-classification', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('setup', type=str, nargs='+', choices=setting_dict.keys(), help='experiment setup')
    parser.add_argument('-m', '--model-name', type=str, nargs='+', default=['resnet50'], help='list of model names')
    parser.add_argument('-t', '--model-type', type=str, default='pic', help='model type')
    parser.add_argument('-c', '--cuda', type=str, default='0', help='cuda device')
    parser.add_argument('-o', '--output-dir', type=str, default='log', help='log dir')
    parser.add_argument('-p', '--project-name', type=str, default='pytorch-image-classification', help='project name used for wandb logger')
    parser.add_argument('-w', '--who', type=str, default='Hankyul', help='enter your name')
    parser.add_argument('--use-wandb', action='store_true', default=False, help='use wandb')

    return parser


def pass_required_variable_from_previous_args(args, prev_args=None):
    if prev_args:
        required_vars = ['gpu', 'world_size', 'distributed', 'is_rank_zero', 'device']
        for var in required_vars:
            exec(f"args.{var} = prev_args.{var}")


def save_arguments(args, is_master):
    if is_master:
        print("Multiple Train Setting")
        print(f" - model (num={len(args.model_name)}): {', '.join(args.model_name)}")
        print(f" - setting (num={len(args.setup)}): {', '.join(args.setup)}")
        print(f" - cuda: {args.cuda}")
        print(f" - output dir: {args.output_dir}")

        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(args.output_dir, 'last_multi_args.txt'), 'wt') as f:
            f.write(" ".join(sys.argv))


if __name__ == '__main__':
    is_master = os.environ.get('LOCAL_RANK', None) is None or int(os.environ['LOCAL_RANK']) == 0
    multi_args_parser = get_multi_args_parser()
    multi_args = multi_args_parser.parse_args()
    save_arguments(multi_args, is_master)
    prev_args = None

    for setup in multi_args.setup:
        args_parser = get_args_parser()
        args = args_parser.parse_args(setting_dict[setup].split(' '))
        pass_required_variable_from_previous_args(args, prev_args)
        for model_name in multi_args.model_name:
            # args.epoch=1
            args.setup = setup
            args.exp_name = f"{model_name}_{setup}"
            args.model_name = model_name
            for option_name in ['model_type', 'cuda', 'output_dir', 'project_name', 'who', 'use_wandb']:
                exec(f"args.{option_name} = multi_args.{option_name}")
            try:
                run(args)
            except:
                print(print_exc())
            clear(args)
        prev_args = args