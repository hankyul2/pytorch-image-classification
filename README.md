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