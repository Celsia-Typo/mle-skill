# Blueprint: Cassava Leaf Disease Classification

> ✅ **Nature of this notebook: Training pipeline with conceptual introduction.**
> Covers ViT architecture explanation, dataset construction, TPU-distributed training,
> and model checkpoint saving. No inference or submission CSV is generated.
> Source file: `cassava-leaf-disease-classification.ipynb`

---

## Competition Details

| Field | Value |
|---|---|
| Competition | [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) |
| Task | 5-class image classification of cassava leaf disease types |
| Input | JPEG images of cassava leaves |
| Target | One of 5 classes: Cassava Bacterial Blight (CBB), Cassava Brown Streak Disease (CBSD), Cassava Green Mottle (CGM), Cassava Mosaic Disease (CMD), Healthy |
| Evaluation | Categorization accuracy |

---

## Dependencies

| Library | Role |
|---|---|
| `timm` | ViT-Base/16 pretrained model + weights loading |
| `torch`, `torchvision` | Model, dataset, transforms, DataLoader |
| `torch_xla` (v1.7) | TPU device management, optimizer step, model saving |
| `torch_xla.distributed.xla_multiprocessing` | Multi-process spawning across 8 TPU cores |
| `torch_xla.distributed.parallel_loader` | TPU-optimized data pipeline (`ParallelLoader`) |
| `sklearn.model_selection` | Stratified train/validation split |
| `PIL`, `numpy`, `pandas` | Image loading, data handling |
| `tqdm`, `matplotlib` | Progress bars, visualization |

Install:
```bash
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version 1.7
!pip install timm
```

---

## Configuration

```python
IMG_SIZE   = 224
BATCH_SIZE = 16
LR         = 2e-05      # per core; scaled by world_size at runtime
GAMMA      = 0.7        # defined but not used (no LR scheduler applied)
N_EPOCHS   = 10

MODEL_PATH = "../input/vit-base-models-pretrained-pytorch/jx_vit_base_p16_224-80ecf9dd.pth"
```

TPU environment flags:
```python
os.environ["XLA_USE_BF16"] = "1"                    # bfloat16 precision on TPU
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"
```

---

## Data

| File | Description |
|---|---|
| `train.csv` | `image_id`, `label` (0–4) |
| `train_images/` | JPEG images of cassava leaves |
| `test_images/` | Test images (not used in this notebook) |

**Split:** 90/10 stratified split by label
```python
train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.1, random_state=42, stratify=df.label.values
)
```

---

## Pipeline

### Step 1 — Dataset

```python
class CassavaDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        img_name, label = self.df_data[index]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(img)
        return image, label
```

### Step 2 — Augmentation

**Training transforms:**
```python
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet stats
])
```

**Validation transforms:** Resize + ToTensor + Normalize only (no augmentation).

### Step 3 — Model: ViTBase16

```python
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)
        if pretrained:
            self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)
```

- **Backbone:** `vit_base_patch16_224` from timm — 12-layer Transformer encoder, patch size 16×16, hidden dim 768, 12 attention heads
- **Input:** 224×224 RGB image → 196 patches of 16×16 each + 1 `[CLS]` token = 197 sequence tokens
- **Head:** original 1000-class ImageNet head replaced by `nn.Linear(768, 5)`
- **Pretrained weights:** loaded from local `.pth` file (offline Kaggle dataset)

The model class also contains `train_one_epoch` and `validate_one_epoch` as instance methods (training logic encapsulated in the model class).

### Step 4 — TPU-Distributed Training Setup

```python
def _run():
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),  # 8 TPU cores
        rank=xm.get_ordinal(),
        shuffle=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=16,
                              sampler=train_sampler, drop_last=True, num_workers=8)

    device = xm.xla_device()
    model.to(device)

    # LR scaled linearly with number of TPU cores
    lr = LR * xm.xrt_world_size()   # 2e-5 × 8 = 1.6e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

### Step 5 — Training Loop

```python
def fit_tpu(model, epochs, device, criterion, optimizer, train_loader, valid_loader):
    for epoch in range(1, epochs + 1):
        para_train_loader = pl.ParallelLoader(train_loader, [device])
        train_loss, train_acc = model.train_one_epoch(
            para_train_loader.per_device_loader(device), criterion, optimizer, device
        )
        para_valid_loader = pl.ParallelLoader(valid_loader, [device])
        valid_loss, valid_acc = model.validate_one_epoch(
            para_valid_loader.per_device_loader(device), criterion, device
        )
```

- **Loss:** `nn.CrossEntropyLoss()`
- **Optimizer:** Adam with linear LR scaling across TPU cores
- **TPU optimizer step:** `xm.optimizer_step(optimizer)` (synchronizes gradients across cores before update)
- **Parallel loading:** `pl.ParallelLoader` wraps the DataLoader to feed each TPU core its own device shard
- **Logging:** `xm.master_print()` ensures only core 0 prints (avoids 8× duplicate output)
- **Accuracy:** computed as `(output.argmax(dim=1) == target).float().mean()` per batch

### Step 6 — Multi-Process Spawn

```python
def _mp_fn(rank, flags):
    torch.set_default_tensor_type("torch.FloatTensor")
    _run()

FLAGS = {}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method="fork")
```

`xmp.spawn` forks 8 processes, one per TPU core, each running `_run()` independently with shared model weights and synchronized gradients via XLA collective operations.

### Step 7 — Model Checkpoint

```python
xm.save(model.state_dict(), f'model_5e_{datetime.now().strftime("%Y%m%d-%H%M")}.pth')
```

Saved at the end of all epochs with a timestamp in the filename. `xm.save` is XLA-safe (equivalent to `torch.save` but handles TPU tensor serialization). Note: mid-training best-model saving is commented out in the `fit_tpu` loop.

---

## Output

| File | Description |
|---|---|
| `model_5e_<timestamp>.pth` | Final model state dict after all epochs |

No inference or `submission.csv` is generated — a separate inference notebook is required.

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| ViT-Base/16 (patch 16×16) | Treats image as a sequence of 196 patches; global self-attention from first layer unlike CNNs with limited early receptive fields |
| Load pretrained weights from local `.pth` | Internet is disabled during Kaggle training; weights loaded from offline dataset |
| Replace only the final head | Transfer learning: freeze backbone representations, fine-tune only the classification head is common but here the full model is fine-tuned |
| LR × world_size scaling | Linear scaling rule for distributed training: effective batch = 16 × 8 = 128; LR should scale accordingly |
| `DistributedSampler` | Ensures each TPU core sees a non-overlapping shard of the training data per epoch |
| `XLA_USE_BF16=1` | bfloat16 on TPU is faster and uses less memory; TPU hardware has native bfloat16 support |
| `drop_last=True` | Ensures all batches have the same size — required for TPU static shape compilation |
| Stratified split by label | Class imbalance in cassava dataset (CMD is dominant); stratification ensures proportional representation in both splits |

---

## ViT Architecture Summary

| Component | Detail |
|---|---|
| Patch size | 16×16 pixels |
| Sequence length | 196 patches + 1 `[CLS]` token = 197 tokens |
| Hidden dimension | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| MLP dimension | 3072 |
| Positional embedding | Learnable 1D embeddings added to patch embeddings |
| Classification | Output of `[CLS]` token → Linear head → 5 classes |

---

## SOTA Gap

| Aspect | This Notebook | Competition SOTA |
|---|---|---|
| Model | ViT-Base/16 (single model) | EfficientNet-B4/B7 ensemble + ViT ensemble |
| Augmentation | HFlip, VFlip, RandomResizedCrop | CutMix, MixUp, GridMask, heavy color jitter |
| Training | Single fold, 10 epochs | 5-fold CV, cosine annealing, label smoothing |
| Image size | 224×224 | 384×512 (larger resolution) |
| Accuracy (valid) | ~85% (baseline) | ~90%+ (top solutions) |

---

## Suggested Improvements

1. **Add stronger augmentation** — CutMix and MixUp are highly effective for this competition; AugMix and GridDistortion also improve robustness
2. **Use larger image resolution** — ViT is sensitive to patch size vs. image resolution; training at 384×384 significantly improves accuracy
3. **Add LR scheduler** — `GAMMA` is defined but never used; cosine annealing or OneCycleLR stabilizes training and avoids overshooting
4. **5-fold cross-validation** — single 90/10 split has high variance; stratified k-fold gives more reliable validation scores
5. **Label smoothing** — cassava labels have known annotation noise; `CrossEntropyLoss(label_smoothing=0.1)` improves generalization
6. **Add submission inference** — this notebook only trains; a companion inference notebook is needed to generate `submission.csv`
7. **Try EfficientNet ensemble** — competition winners combined ViT and EfficientNet predictions; both architectures have complementary error patterns
8. **Save best checkpoint mid-training** — the `xm.save` inside `fit_tpu` is commented out; restoring the best-epoch weights instead of the final-epoch weights typically improves validation accuracy
