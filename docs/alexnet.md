# AlexNet

AlexNet Summary Page



## Experiment

Experiment AlexNet on FashionMnist, CIFAR10, CIFAR100, ImageNet



### Fashion MNIST

AlexNet on FashionMNIST (epoch=20, top-1 acc=94.38)

```bash
python3 train.py data --dataset_type FashionMNIST --train-size 227 227 --train-resize-mode ResizeRandomCrop --interpolation bicubic --random-crop-pad 4 --test-size 227 227 --center-crop-ptr 1.0 --mean 0.1307 --std 0.3081 --cutmix 0.0 --mixup 0.0 --remode 0.2 --cuda 4 -m alexnet --model-type pic --in-channels 1 --dropout 0.1 --smoothing 0.1 --epoch 20 --weight-decay 1e-4 --scheduler onecyclelr -b 512 -j 16 --pin-memory --amp --channels-last
```

