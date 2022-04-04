# pytorch-image-classification
 Pytorch Image Classification. This is just toy example for studying. Most code is ported from timm.



## Tutorial
How to run this?
```bash
torchrun --nproc_per_node=2 train.py imageNet --cuda 7,8 
```

```bash
python3 train.py imageNet --cuda 7 
```

What does each variable mean?
- nproc_per_node - control parallelism
- cuda - control which gpu device to use



## Need to remember
```bash
torchrun --nproc_per_node=2 train.py imageNet --interpolation bicubic --lr 1e-3 --epoch 50 --warmup-lr 1e-6 -j 8 --pin-memory --amp --channels-last --cuda 7,8 --who hankyul --exp-target model_name
```

1. You should choose experiment_target or set experiment name manually.
2. You should specify your name & project name
3. resize operation in test time is done like this: `(H, W) -> resize -> (test_size/crop_ptr, test_size/crop_ptr) -> center_crop -> (test_size, test_size)`. It makes me confused at first. But you can set test_image size directly.



## Issue
1. Validation scores are not same w/ w/o distributed mode. This is because `len(dataset) % (batch_size * ngpu) != 0`. Batch sizes for each gpu's last iter are different and averaging them together results in different validation score. Difference is about `0.04%`.