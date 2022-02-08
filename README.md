# pytorch-image-classification
 pytorch image classification

## Tutorial
How to run this?
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 pt_elastic.py
```

What does each variable mean?
- CUDA_DEVICE_ORDER - control device order
- CUDA_VISIBLE_DEVICES - control device id
- nproc_per_node - control parallelism
- device - control between cpu and gpu