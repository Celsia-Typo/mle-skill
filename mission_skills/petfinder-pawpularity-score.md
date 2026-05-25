# Petfinder Pawpularity Score — SCoT Prompt Guide

## Part 1: Natural Language Instructions

You are solving a Kaggle computer vision + tabular regression problem: **PetFinder Pawpularity Score**.

**Task**: Given a pet photo (JPEG) and 12 binary tabular features, predict the `Pawpularity` score (integer 0–100). Metric is RMSE on the raw 0–100 scale.

> ⚠️ **Distribution shift warning**: Training labels are single-rater integer scores; test labels are multi-rater averages. This makes the test distribution smoother and more centred than the training distribution. Local CV RMSE will therefore be **optimistic** relative to the true leaderboard score. The primary mitigation is Gaussian soft labels (see item 1 below).

**Recommended strategy**:
1. **Classification mode with Gaussian soft labels** (101 classes, 0–100): **Do NOT use hard CrossEntropyLoss or label smoothing.** For each integer label `y`, build a Gaussian soft-label vector: `soft[i] = exp(-(i - y)² / (2 · σ²))` for i in 0..100, then normalise to sum=1 (use **σ = 3.0**). Train with `F.kl_div(F.log_softmax(logits, dim=1), soft, reduction='batchmean')`. This directly bridges the train/test distribution gap. At inference, convert soft probabilities to a scalar via expected value: `(softmax(logits) * arange(101)).sum(1)`.
2. **Backbone — choose one per runfile for diversity** (runfiles MUST use architectures from **different families** to maximise ensemble gain — do NOT use two Swin variants):
   - **Option A** (ViT / Swin family): `swin_large_patch4_window12_384_in22k`, feat_dim=1536, resize=384, batch_size=16
   - **Option B** (pure CNN family): `convnextv2_large.fcmae_ft_in22k_in1k_384`, feat_dim=1536, resize=384, batch_size=16
   - **Option C** (EVA / CLIP-pretrained family): `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`, feat_dim=1024, resize=448, batch_size=8

   Strip the backbone head with `timm.create_model(..., pretrained=True, num_classes=0, global_pool='avg')`. Then add `nn.Dropout(0.3)` followed by a single `nn.Linear(feat_dim + 12, 101)` head.

   **⚠️ Pretrained weights**: `pretrained=True` uses cached weights from `~/.cache/torch/hub/` — do NOT download at runtime. Always wrap model creation in a try/except and fall back to `pretrained=False` if needed.

3. **5-fold CV**: split with `StratifiedKFold(n_splits=5)` on a binned version of the target (`pd.cut(Pawpularity, bins=10)`). Train each fold, save the best checkpoint by validation RMSE.
4. **Training recipe**: AMP (`autocast` + `GradScaler`), gradient clipping (`max_norm=1000`), **layer-wise `AdamW`** with two param groups — backbone lr=1e-5, head lr=1e-4 — and weight_decay=1e-5. Use `CosineAnnealingLR(T_max=max_epochs, eta_min=1e-6)`. **Early stopping with patience=3**, max_epochs=20.
5. **Data augmentation (train)**:
   - Standard: random horizontal flip, random rotation ±10°, ColorJitter (brightness=0.2, contrast=0.2), resize to target resolution, ImageNet normalisation, **RandomErasing(p=0.5)** after ToTensor.
   - **Mixup** (alpha=0.4): with probability 0.5 per batch, blend two samples linearly; mix both the images **and** their Gaussian soft-label vectors: `mixed_soft = lam · soft_a + (1-lam) · soft_b`, then compute KL div against the mixed soft label.
   - Validation/test: resize only + normalise (no augmentation).
6. **TTA (Test-Time Augmentation)**: At inference on the test set, run each image through both the original and the horizontally-flipped version. Average the two softmax probability distributions **before** computing the expected value. Apply TTA also for validation RMSE during training to keep the metric consistent. This reliably reduces RMSE by ~0.1–0.2 at zero training cost.
7. **Memory safety**: use `torch.no_grad()` + `autocast` during inference; iterate in batches; delete model + call `gc.collect()` + `torch.cuda.empty_cache()` after each fold.

Write the rough problem-solving steps using sequential, branch, and loop structures first, then output the final code.

---

## Part 2: SCoT Demonstration Examples

### Example 1 — `PetDataset.__getitem__`

```python
def __getitem__(self, idx):
    """
    Load one sample (image + tabular features + label).
    Input: idx: int
    Output: (img_tensor: Tensor[3,H,W], tab: Tensor[12], label: Tensor[]) for train/val
            (img_tensor: Tensor[3,H,W], tab: Tensor[12])                   for test
    """
    # SEQUENTIAL: build image path from Id column
    # SEQUENTIAL: open with PIL → convert RGB (fallback: black image on exception)
    # SEQUENTIAL: apply self.transform → img_tensor
    # SEQUENTIAL: read 12 META_COLS → float32 tensor
    # BRANCH: if self.is_test → return (img, tab)
    # BRANCH: else → label = int64(Pawpularity); return (img, tab, label)
    pass
```

```python
def __getitem__(self, idx):
    row = self.df.iloc[idx]
    try:
        img = Image.open(self.img_dir / f"{row['Id']}.jpg").convert("RGB")
    except Exception:
        img = Image.fromarray(np.zeros((self.resize, self.resize, 3), dtype=np.uint8))
    if self.transform:
        img = self.transform(img)
    tab = torch.tensor(row[META_COLS].values.astype(np.float32))
    if self.is_test:
        return img, tab
    label = torch.tensor(int(row["Pawpularity"]), dtype=torch.long)
    return img, tab, label
```

---

### Example 2 — Gaussian soft label construction

```python
def make_soft_labels(labels_int, num_classes=101, sigma=3.0):
    """
    Convert integer Pawpularity labels to Gaussian soft-label distributions.
    Motivation: test labels are multi-rater averages (smoother distribution);
                Gaussian soft labels bridge the train/test distribution gap.
    Input:  labels_int: Tensor[B] int64, values in 0–100
            num_classes: int = 101
            sigma: float = 3.0
    Output: soft: Tensor[B, 101] float32, each row sums to 1
    """
    # SEQUENTIAL: build class index range r → shape [1, 101]
    # SEQUENTIAL: expand labels_int → [B, 1]; compute squared distances → [B, 101]
    # SEQUENTIAL: apply Gaussian kernel: exp(-(r - y)^2 / (2 * sigma^2))
    # SEQUENTIAL: normalise each row so it sums to 1
    pass
```

```python
def make_soft_labels(labels_int, num_classes=101, sigma=3.0):
    r = torch.arange(num_classes, dtype=torch.float32, device=labels_int.device)  # [101]
    y = labels_int.float().unsqueeze(1)           # [B, 1]
    dist_sq = (r.unsqueeze(0) - y) ** 2           # [B, 101]
    soft = torch.exp(-dist_sq / (2 * sigma ** 2))
    return soft / soft.sum(dim=1, keepdim=True)   # [B, 101], normalised
```

---

### Example 3 — TTA inference

```python
def inference_fn_tta(model, loader, cfg):
    """
    Batched inference with horizontal-flip TTA.
    Input:  model, loader (test or val DataLoader), cfg
    Output: probs: ndarray[N, num_classes]  (softmax probabilities averaged over orig + hflip)
    """
    # SEQUENTIAL: model.eval(); initialise all_probs list
    # LOOP: for each batch in loader
    #   SEQUENTIAL: imgs, tabs = batch[0], batch[1] → move to device
    #   SEQUENTIAL: imgs_flip = torch.flip(imgs, dims=[3])   (horizontal flip)
    #   SEQUENTIAL: with autocast: probs_orig = softmax(model(imgs, tabs))
    #   SEQUENTIAL: with autocast: probs_flip = softmax(model(imgs_flip, tabs))
    #   SEQUENTIAL: probs_avg = (probs_orig + probs_flip) / 2 → append to all_probs
    # SEQUENTIAL: return np.concatenate(all_probs)
    pass
```

```python
@torch.no_grad()
def inference_fn_tta(model, loader, cfg):
    model.eval()
    all_probs = []
    for batch in loader:
        imgs, tabs = batch[0].to(cfg.device), batch[1].to(cfg.device)
        imgs_flip = torch.flip(imgs, dims=[3])          # horizontal flip
        with autocast(enabled=cfg.use_amp):
            probs_orig = torch.softmax(model(imgs,      tabs), dim=1)
            probs_flip = torch.softmax(model(imgs_flip, tabs), dim=1)
        probs_avg = (probs_orig + probs_flip) / 2
        all_probs.append(probs_avg.cpu().numpy())
    return np.concatenate(all_probs, axis=0).astype(np.float32)
```

---

### Example 4 — Layer-wise LR fold training with early stopping

```python
def train_fold(model, trn_loader, val_loader, cfg, fold_id):
    """
    Train one fold: layer-wise LR, Gaussian soft labels, TTA val RMSE, early stopping.
    Input:  model, trn_loader, val_loader,
            cfg: has max_epochs, patience, backbone_lr, head_lr, weight_decay,
                 grad_clip, use_amp, num_classes, soft_label_sigma
    Output: (best_rmse: float, ckpt_path: str)
    """
    # SEQUENTIAL: AdamW with two param groups → backbone_lr for backbone, head_lr for head
    # SEQUENTIAL: CosineAnnealingLR(T_max=max_epochs, eta_min=1e-6), GradScaler
    # SEQUENTIAL: best_rmse = inf; patience_counter = 0
    # LOOP: for epoch in range(cfg.max_epochs)
    #   SEQUENTIAL: train_one_epoch (KL div + Gaussian soft labels + Mixup) → trn_loss
    #   SEQUENTIAL: inference_fn_tta on val_loader → val_probs → expected_value → rmse
    #   BRANCH: if rmse < best_rmse
    #     SEQUENTIAL: best_rmse = rmse; patience_counter = 0; torch.save(model, ckpt)
    #   BRANCH: else
    #     SEQUENTIAL: patience_counter += 1
    #     BRANCH: if patience_counter >= cfg.patience → break
    #   SEQUENTIAL: scheduler.step()
    # SEQUENTIAL: return best_rmse, ckpt_path
    pass
```

```python
def train_fold(model, trn_loader, val_loader, cfg, fold_id):
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg.backbone_lr},
        {'params': list(model.dropout.parameters()) + list(model.head.parameters()),
         'lr': cfg.head_lr},
    ], weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=cfg.use_amp)
    r = np.arange(cfg.num_classes)

    best_rmse, patience_counter = 1e9, 0
    ckpt_path = f"petnet_fold{fold_id}.pth"

    for epoch in range(cfg.max_epochs):
        trn_loss  = train_one_epoch(model, trn_loader, optimizer, scaler, cfg)
        val_probs = inference_fn_tta(model, val_loader, cfg)   # [N_val, 101] with TTA
        val_preds = (val_probs * r).sum(axis=1)
        rmse      = np.sqrt(np.mean((val_preds - val_loader.dataset.labels) ** 2))
        print(f"  Epoch {epoch+1:02d} | loss={trn_loss:.4f} | val_rmse={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ checkpoint (rmse={rmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Early stop at epoch {epoch+1}")
                break
        scheduler.step()

    return best_rmse, ckpt_path
```

---

## Part 3: Complete Solution Scaffold

```python
# ── petfinder_pawpularity.py ─────────────────────────────────────────────────
import os, random, gc
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms as T
from sklearn.model_selection import StratifiedKFold

# ── Backbone registry ──────────────────────────────────────────────────────────
# Choose ONE per runfile — MUST be from different architecture families:
#   OPTION_A: ('swin_large_patch4_window12_384_in22k',             1536, 384, 16)  ← ViT/Swin
#   OPTION_B: ('convnextv2_large.fcmae_ft_in22k_in1k_384',         1536, 384, 16)  ← pure CNN
#   OPTION_C: ('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',   1024, 448,  8)  ← EVA/CLIP
BACKBONE, FEAT_DIM, RESIZE, BATCH_SIZE = (
    'swin_large_patch4_window12_384_in22k', 1536, 384, 16   # ← change per runfile
)

META_COLS = ['Subject Focus','Eyes','Face','Near','Action',
             'Accessory','Group','Collage','Human','Occlusion','Info','Blur']

# ── Config ───────────────────────────────────────────────────────────────────
class Config:
    seed             = 42
    num_classes      = 101
    model_name       = BACKBONE
    feat_dim         = FEAT_DIM
    resize           = RESIZE
    batch_size       = BATCH_SIZE
    backbone_lr      = 1e-5          # lower LR for pretrained backbone
    head_lr          = 1e-4          # higher LR for new dropout + head
    weight_decay     = 1e-5
    dropout_rate     = 0.3
    soft_label_sigma = 3.0           # Gaussian sigma — bridges train/test dist. gap
    max_epochs       = 20            # early stopping will trigger before this
    patience         = 3
    n_splits         = 5
    grad_clip        = 1000.0
    mixup_alpha      = 0.4
    use_amp          = True
    device           = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir        = "./input/train"
    test_dir         = "./input/test"

def seed_everything(seed):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Augmentation ──────────────────────────────────────────────────────────────
def get_transforms(phase, resize):
    # BRANCH: train → flip + rotation + jitter + RandomErasing
    # BRANCH: val/test → resize + normalise only
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    if phase == 'train':
        return T.Compose([
            T.Resize((resize, resize)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.5, scale=(0.02, 0.33)),
        ])
    return T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────
class PetDataset(Dataset):
    def __init__(self, df, img_dir, phase='test', cfg=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.phase     = phase
        self.resize    = cfg.resize if cfg else 384
        self.transform = get_transforms(phase, self.resize)
        self.tabular   = df[META_COLS].values.astype(np.float32)
        if phase != 'test':
            self.labels = df['Pawpularity'].values.astype(np.float32)   # for RMSE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # SEQUENTIAL: load image with PIL fallback; apply transform
        row = self.df.iloc[idx]
        try:
            img = Image.open(self.img_dir / f"{row['Id']}.jpg").convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((self.resize, self.resize, 3), dtype=np.uint8))
        img = self.transform(img)
        tab = torch.tensor(self.tabular[idx])
        # BRANCH: test → no label; train/val → int64 label (soft labels built in training loop)
        if self.phase == 'test':
            return img, tab
        label = torch.tensor(int(row['Pawpularity']), dtype=torch.long)
        return img, tab, label


# ── Gaussian soft labels ──────────────────────────────────────────────────────
def make_soft_labels(labels_int, num_classes=101, sigma=3.0):
    # SEQUENTIAL: compute per-class Gaussian weights centred at each label
    # SEQUENTIAL: normalise rows to sum=1
    r = torch.arange(num_classes, dtype=torch.float32, device=labels_int.device)
    y = labels_int.float().unsqueeze(1)
    soft = torch.exp(-((r.unsqueeze(0) - y) ** 2) / (2 * sigma ** 2))
    return soft / soft.sum(dim=1, keepdim=True)


# ── Model ─────────────────────────────────────────────────────────────────────
class PetNet(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or Config()
        # BRANCH: pretrained=True uses hub cache; fall back to False on error
        try:
            self.backbone = timm.create_model(
                cfg.model_name, pretrained=True, num_classes=0, global_pool='avg')
        except Exception:
            self.backbone = timm.create_model(
                cfg.model_name, pretrained=False, num_classes=0, global_pool='avg')
        for attr in ('head', 'head_dist'):
            if hasattr(self.backbone, attr):
                setattr(self.backbone, attr, nn.Identity())
        in_dim       = cfg.feat_dim + len(META_COLS)
        self.dropout = nn.Dropout(cfg.dropout_rate)   # regularise before head
        self.head    = nn.Linear(in_dim, cfg.num_classes)

    def forward(self, x, tab):
        # SEQUENTIAL: extract features; handle tuple/4D/3D edge cases
        feat = self.backbone(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        if feat.dim() == 4:
            if feat.shape[-1] == self.backbone.num_features:
                feat = feat.mean(dim=[1, 2])   # channels-last  → [B, C]
            else:
                feat = feat.mean(dim=[2, 3])   # channels-first → [B, C]
        elif feat.dim() == 3:
            feat = feat.mean(dim=1)
        # SEQUENTIAL: concatenate tabular → dropout → classification head
        x = torch.cat([feat, tab], dim=1)
        return self.head(self.dropout(x))      # [B, 101]


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, cfg):
    model.train()
    total_loss = 0.0
    for imgs, tabs, labels in loader:
        imgs, tabs, labels = imgs.to(cfg.device), tabs.to(cfg.device), labels.to(cfg.device)
        optimizer.zero_grad()
        with autocast(enabled=cfg.use_amp):
            # SEQUENTIAL: generate Gaussian soft labels for this batch
            soft = make_soft_labels(labels, cfg.num_classes, cfg.soft_label_sigma)
            # BRANCH: apply Mixup ~50% of the time; mix images, tabs, AND soft labels
            if np.random.rand() < 0.5:
                lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
                idx  = torch.randperm(imgs.size(0), device=imgs.device)
                imgs = lam * imgs + (1 - lam) * imgs[idx]
                tabs = lam * tabs + (1 - lam) * tabs[idx]
                soft = lam * soft + (1 - lam) * soft[idx]
            logits = model(imgs, tabs)
            loss   = F.kl_div(F.log_softmax(logits, dim=1), soft, reduction='batchmean')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Inference with TTA ────────────────────────────────────────────────────────
@torch.no_grad()
def inference_fn_tta(model, loader, cfg):
    # OUTPUT: ndarray[N, num_classes]  (softmax probs averaged over orig + hflip)
    model.eval()
    all_probs = []
    for batch in loader:
        imgs, tabs = batch[0].to(cfg.device), batch[1].to(cfg.device)
        imgs_flip  = torch.flip(imgs, dims=[3])
        with autocast(enabled=cfg.use_amp):
            probs_orig = torch.softmax(model(imgs,      tabs), dim=1)
            probs_flip = torch.softmax(model(imgs_flip, tabs), dim=1)
        all_probs.append(((probs_orig + probs_flip) / 2).cpu().numpy())
    return np.concatenate(all_probs, axis=0).astype(np.float32)


# ── Cross-validation ──────────────────────────────────────────────────────────
def run_cv(train_df, cfg):
    bins      = pd.cut(train_df['Pawpularity'], bins=10, labels=False)
    skf       = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    oof_preds = np.zeros(len(train_df))
    r         = np.arange(cfg.num_classes)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, bins)):
        print(f"\n===== Fold {fold+1}/{cfg.n_splits} =====")
        trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)

        trn_ds     = PetDataset(trn_df, cfg.train_dir, phase='train', cfg=cfg)
        val_ds     = PetDataset(val_df, cfg.train_dir, phase='val',   cfg=cfg)
        trn_loader = DataLoader(trn_ds, batch_size=cfg.batch_size,
                                shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

        model = PetNet(cfg).to(cfg.device)
        # SEQUENTIAL: layer-wise LR — backbone uses backbone_lr, head uses head_lr
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': cfg.backbone_lr},
            {'params': list(model.dropout.parameters()) + list(model.head.parameters()),
             'lr': cfg.head_lr},
        ], weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epochs, eta_min=1e-6)
        scaler = GradScaler(enabled=cfg.use_amp)

        best_rmse, patience_counter = 1e9, 0
        ckpt_path = f"petnet_fold{fold}.pth"

        # LOOP: epoch loop with early stopping
        for epoch in range(cfg.max_epochs):
            trn_loss  = train_one_epoch(model, trn_loader, optimizer, scaler, cfg)
            val_probs = inference_fn_tta(model, val_loader, cfg)   # [N_val, 101]
            val_preds = (val_probs * r).sum(axis=1)
            rmse      = np.sqrt(np.mean((val_preds - val_df['Pawpularity'].values) ** 2))
            print(f"  Epoch {epoch+1:02d} | loss={trn_loss:.4f} | val_rmse={rmse:.4f}")

            # BRANCH: save checkpoint if improved; else increment patience counter
            if rmse < best_rmse:
                best_rmse = rmse
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ✓ checkpoint saved (rmse={rmse:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            scheduler.step()

        # SEQUENTIAL: reload best checkpoint for OOF predictions
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.to(cfg.device)
        val_probs = inference_fn_tta(model, val_loader, cfg)
        oof_preds[val_idx] = (val_probs * r).sum(axis=1)

        del model; gc.collect(); torch.cuda.empty_cache()

    oof_rmse = np.sqrt(np.mean((oof_preds - train_df['Pawpularity'].values) ** 2))
    print(f"\nOOF RMSE: {oof_rmse:.4f}  (note: may be optimistic vs LB due to distribution shift)")
    return oof_preds


# ── Test inference ────────────────────────────────────────────────────────────
def run_test_inference(test_df, cfg):
    test_ds     = PetDataset(test_df, cfg.test_dir, phase='test', cfg=cfg)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    r         = np.arange(cfg.num_classes)
    preds_avg = np.zeros(len(test_df))
    model     = PetNet(cfg).to(cfg.device)

    # LOOP: load each fold checkpoint; accumulate TTA expected-value predictions
    for fold in range(cfg.n_splits):
        model.load_state_dict(torch.load(f"petnet_fold{fold}.pth", map_location='cpu'))
        model.to(cfg.device)
        raw = inference_fn_tta(model, test_loader, cfg)   # [N, 101]
        preds_avg += (raw * r).sum(axis=1)

    preds_avg /= cfg.n_splits
    return np.clip(preds_avg, 0, 100)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg = Config()
    seed_everything(cfg.seed)

    train_df = pd.read_csv("./input/train.csv")
    test_df  = pd.read_csv("./input/test.csv")

    oof_preds  = run_cv(train_df, cfg)
    test_preds = run_test_inference(test_df, cfg)

    submission = pd.DataFrame({'Id': test_df['Id'], 'Pawpularity': test_preds})
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv"); print(submission.head())


if __name__ == "__main__":
    main()
```

---

## Part 4: Trigger Prompts

Now implement the complete solution for the **PetFinder Pawpularity Score** task.

- **Distribution shift**: Training labels are single-rater integers; test labels are multi-rater averages. Local CV RMSE is optimistic. Use **Gaussian soft labels (σ=3)** with KL divergence loss to bridge this gap — do NOT use hard CrossEntropyLoss.
- **Choose a backbone** from the registry (one per runfile — architectures MUST come from different families):
  - Option A: `swin_large_patch4_window12_384_in22k` (feat_dim=1536, resize=384, batch=16) — ViT/Swin
  - Option B: `convnextv2_large.fcmae_ft_in22k_in1k_384` (feat_dim=1536, resize=384, batch=16) — pure CNN
  - Option C: `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` (feat_dim=1024, resize=448, batch=8) — EVA/CLIP
- Add `nn.Dropout(0.3)` before the final linear head.
- Use **layer-wise AdamW**: backbone lr=1e-5, head lr=1e-4, weight_decay=1e-5.
- Train with `StratifiedKFold(n_splits=5)`, AMP, `CosineAnnealingLR`, **early stopping patience=3**, max_epochs=20.
- Apply **Mixup (alpha=0.4)** with 50% probability per batch — mix images, tabs, **and Gaussian soft-label vectors**.
- Apply **RandomErasing(p=0.5)** in the train transform pipeline.
- At inference, use **horizontal-flip TTA**: average softmax distributions of original + flipped image, then compute expected value `(probs * arange(101)).sum(1)`.
- Average predictions across 5 folds, clip to [0, 100], and write `submission.csv`.

Let's think step by step and write your code here.
