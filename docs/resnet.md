# ResNet

This is ResNet Summary Page.





## Experiment

Experiment resnet50, 101 on FashionMNIST, CIFAR10, CIFAR100, ImageNet



### Fashion MNIST

ResNet50 on FashionMNIST (gpu=1, epoch=10, top-1 acc=93.52)

```bash
python3 train.py data --dataset_type FashionMNIST --train-resize-mode ResizeRandomCrop --random-crop-pad 4 --center-crop-ptr 1.0 --mean 0.1307 --std 0.3081 --cutmix 0.0 --mixup 0.0 --remode 0.0 --cuda 5 -m resnet50 --in-channels 1 --drop-path-rate 0.1 --smoothing 0.0 --epoch 10 --weight-decay 1e-4 --scheduler onecyclelr -b 512 -j 16 --pin-memory --amp --channels-last
```



ResNet50 on FashionMNIST (gpu=2, epoch=10, top-1 acc=93.61)

```bash
torchrun --nproc_per_node=2 train.py data --dataset_type FashionMNIST --train-resize-mode ResizeRandomCrop --random-crop-pad 4 --center-crop-ptr 1.0 --mean 0.1307 --std 0.3081 --cutmix 0.0 --mixup 0.0 --remode 0.0 --cuda 3,4 -m resnet50 --in-channels 1 --drop-path-rate 0.1 --smoothing 0.0 --epoch 10 --lr 1e-3 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last
```



ResNet50 on FashionMNIST (gpu=1, epoch=20, top-1 acc=94.71)

```bash
python3 train.py data --dataset_type FashionMNIST --train-resize-mode ResizeRandomCrop --interpolation bicubic --random-crop-pad 4 --center-crop-ptr 1.0 --mean 0.1307 --std 0.3081 --cutmix 0.0 --mixup 0.0 --remode 0.2 --cuda 5 -m resnet50 --in-channels 1 --drop-path-rate 0.1 --smoothing 0.1 --epoch 20 --weight-decay 1e-4 --scheduler onecyclelr -b 512 -j 16 --pin-memory --amp --channels-last
```



ResNet50 on FashionMNIST (gpu=2, epoch=20, top-1 acc=94.69)

```bash
torchrun --nproc_per_node=2 train.py data --dataset_type FashionMNIST --train-resize-mode ResizeRandomCrop --random-crop-pad 4 --center-crop-ptr 1.0 --mean 0.1307 --std 0.3081 --interpolation bicubic --cutmix 0.0 --mixup 0.0 --remode 0.2 --cuda 3,4 -m resnet50 --in-channels 1 --drop-path-rate 0.1 --smoothing 0.1 --epoch 20 --lr 1e-3 --weight-decay 1e-4 --scheduler onecyclelr -b 256 -j 16 --pin-memory --amp --channels-last
```



ResNet50 on FashionMNIST (gpu=1, epoch=40, top-1 acc=95.21)

```bash
python3 train.py data --dataset_type FashionMNIST --train-resize-mode ResizeRandomCrop --interpolation bicubic --random-crop-pad 4 --center-crop-ptr 1.0 --mean 0.1307 --std 0.3081 --cutmix 0.0 --mixup 0.0 --remode 0.5 --cuda 5 -m resnet50 --in-channels 1 --drop-path-rate 0.1 --smoothing 0.1 --epoch 40 --weight-decay 1e-4 --scheduler onecyclelr -b 512 -j 16 --pin-memory --amp --channels-last
```



