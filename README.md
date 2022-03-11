# pytorch-image-classification
 pytorch image classification

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

## Issue
1. Validation scores are not same w/ w/o distributed mode. This is because `len(dataset) % (batch_size * ngpu) != 0`. 
Batch sizes for each gpu's last iter are different and averaging them together results in different validation score.
Difference is about `0.04%`.