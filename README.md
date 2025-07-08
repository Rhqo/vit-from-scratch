# Vision Transformer from Scratch

## `train.py`

ViT - classification training code

**Usage:**

```bash
uv run train.py
```

**Result:**

```bash
DataLoader: (<torch.utils.data.dataloader.DataLoader object at 0x7ccb043370e0>, <torch.utils.data.dataloader.DataLoader object at 0x7ccc31566bd0>)
Length of train_loader: 1563 batches of 32...
Length of test_loader: 313 batches of 32...
Training Progress (Epoch 1/10): 100%|██████████| 1563/1563 [00:15<00:00, 98.01it/s]
Epoch: 1/10, Train loss: 1.6373, Train acc: 0.4094%, Test acc: 0.4899
...
Training Progress (Epoch 10/10): 100%|██████████| 1563/1563 [00:16<00:00, 97.11it/s]
Epoch: 10/10, Train loss: 0.6396, Train acc: 0.7699%, Test acc: 0.6019
```

## `inference.py`

**Usage:** 

`--infer`: custum model로 inference (9 sampled data)
```bash
uv run inference.py --infer
```

`--eval`: custom model로 test set에 mAP 평가
```bash
uv run inference.py --eval 
```