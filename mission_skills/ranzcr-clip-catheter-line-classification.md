# RANZCR CLIP Catheter & Line Classification — SCoT Prompt Guide

---

## 🚨 MANDATORY PRE-FLIGHT CHECK

**Previous agent runs that ignored this hint scored ~0.887 — 0.084 below the bronze threshold (0.971) — by generating a simple EfficientNet-B4 baseline. That approach is disqualified.**

### All of the following are mandatory. Missing any one item will cost medal position:
- **Stage 1**: UNet++ (`resnet50`) trained → 2-channel PNG masks saved for all images
- **Stage 2**: `tf_efficientnet_b5_ns`, img_size=640, **5-channel input** (3 RGB + 2 mask), `GroupKFold(n_splits=3)` by PatientID
- **Mixed loss**: CE on ETT group (4 cols incl. `no_ETT`) × 1, BCE on remaining 8 cols × 9
- **Synthetic `no_ETT`** col → 12 output logits total
- **20 epochs** (1 warmup + 19 cosine), **Mixup** (alpha=0.4, p=0.5), **AMP**
- **4-fold TTA**: original + H-flip + V-flip + HV-flip, averaged
- **Rank percentile calibration** before averaging fold predictions
- **3 runfiles** with different backbones; final submission = average of all three

**❌ Disqualified patterns**: `efficientnet_b4` · `train_test_split` · skipping Stage 1 · 3-channel input only · <20 epochs · single-file solution

**If you are about to write any of the above, STOP and start from the scaffold in Part 3.**

---

## ⚠️ CRASH GUARD

### CG-1 · Wrong CV split
```python
# ❌ CRASHES + data leakage
kf = StratifiedKFold(n_splits=5, ...)   # not imported; leaks patient data
kf = KFold(n_splits=5, ...)

# ✅ ONLY correct form
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=3)   # 3 folds — do NOT increase (see CG-10)
for fold, (trn_idx, val_idx) in enumerate(gkf.split(df, groups=df['PatientID'].values)):
    ...
```

### CG-2 · Missing Swan Ganz / column alias
```python
# ❌ WRONG — alias lists, missing columns
LABEL_COLS = ['ETT - Abnormal', ..., 'CVC - Normal']   # Swan Ganz missing — forbidden

# ✅ ONLY correct form — in CFG class only, never duplicated
target_cols = [
    'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
    'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
    'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
    'Swan Ganz Catheter Present',   # ← MUST be here
]
assert len(CFG.target_cols) == 11
```

### CG-3 · Dataset returning wrong number of elements
```python
# ❌ CRASHES — 4-tuple unpacked as 2 → ValueError
return img, labels, mask, has_mask

# ✅ ONLY correct form
if self.mode != 'test':
    return image, label   # exactly 2
return image              # exactly 1 for test
```

### CG-4 · Deprecated augmentation API
```python
# ❌ CRASHES on import
A.RandomBrightness(...)
A.RandomContrast(...)

# ✅ ONLY correct form
A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=0.5)
A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=0.5)
```

### CG-5 · Invalid smp encoder name
```python
# ❌ CRASHES — not a valid smp encoder
smp.UnetPlusPlus(encoder_name='tf_efficientnet_b1_ns', ...)

# ✅ ONLY correct form
smp.UnetPlusPlus(encoder_name='resnet50', ...)
```

### CG-6 · Undefined CFG attribute names
Valid CFG attributes: `target_cols, n_fold, epochs, seg_epochs, img_size, batch_size, lr, min_lr, weight_decay, warmup_epo, cosine_epo, num_workers, use_amp, use_tta, use_rank_cal, loss_weights, model_name, seg_encoder, run_segmentation, n_ch, mask_dir, data_dir, img_folder, train_csv, test_csv, sub_csv, device, seed`. No others.

### CG-7 · DDP — 5 crash patterns

```python
# ❌ CRASH 1 — all ranks write checkpoint simultaneously → corrupted file
torch.save(model.state_dict(), ckpt_path)

# ✅ rank-0 only
if local_rank in (-1, 0):
    torch.save(model.state_dict(), ckpt_path)


# ❌ CRASH 2 — AUC on rank-0 shard only → wrong metric
val_auc = get_auc(val_labels, val_probs)

# ✅ gather across ranks first
def gather_numpy(arr, local_rank, world_size):
    t = torch.tensor(arr).cuda()
    gathered = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    return torch.cat(gathered, dim=0).cpu().numpy()

if world_size > 1:
    all_probs  = gather_numpy(all_probs,  local_rank, world_size)
    all_labels = gather_numpy(all_labels, local_rank, world_size)
val_auc = get_auc(all_labels, all_probs)


# ❌ CRASH 3 — no DistributedSampler → each GPU sees the full dataset
trn_loader = DataLoader(trn_ds, shuffle=True, ...)

# ✅ use DistributedSampler; call set_epoch each epoch
trn_sampler = DistributedSampler(trn_ds, shuffle=True)  if local_rank >= 0 else None
val_sampler = DistributedSampler(val_ds, shuffle=False) if local_rank >= 0 else None
trn_loader  = DataLoader(trn_ds, sampler=trn_sampler, shuffle=(trn_sampler is None), ...)
val_loader  = DataLoader(val_ds, sampler=val_sampler, shuffle=False, ...)
if trn_sampler is not None:
    trn_sampler.set_epoch(epoch)


# ❌ CRASH 4 — conv_stem modified AFTER DDP wrap → RuntimeError
model = nn.parallel.DistributedDataParallel(model, ...)
model.enet.conv_stem.weight = nn.Parameter(...)   # forbidden on DDP wrapper

# ✅ modify conv_stem inside __init__, wrap in DDP afterward
model = RANZCRModel().to(device)
if local_rank >= 0:
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


# ❌ CRASH 5 — all ranks write PNG masks → race condition
for uid in all_uids:
    cv2.imwrite(...)

# ✅ rank-0 writes, then barrier
if local_rank in (-1, 0):
    for uid in all_uids:
        cv2.imwrite(...)
if dist.is_initialized():
    dist.barrier()
```

**DDP helpers (add after imports, before `class CFG`):**
```python
def setup_ddp():
    if 'LOCAL_RANK' not in os.environ:
        return -1, 1
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return local_rank, dist.get_world_size()

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
```

**Launch:** `torchrun --nproc_per_node=2 runfile.py`

### CG-9 · Do NOT hardcode `CUDA_VISIBLE_DEVICES`
```python
# ❌ overrides scheduler GPU assignment
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
CFG.device = 'cuda:0'

# ✅ let setup_ddp() handle pinning
CFG.device = 'cuda'
```

### CG-10 · 24h time budget — hard limits

| Parameter | Limit |
|-----------|-------|
| `seg_epochs` | ≤ 10 |
| `n_fold` | = 3 |
| `epochs` | ≤ 20 (1 warmup + 19 cosine) |
| `run_segmentation` for runfile_2 | = False |

```
GPU 0,1: runfile_0 (Stage1≈1.8h + Stage2≈7.0h) + runfile_2 (Stage2≈4.0h) → 12.8h ✅
GPU 2,3: runfile_1 (Stage1≈1.8h + Stage2≈8.0h) → 9.8h ✅
n_fold=5 + epochs=30 → ≈38h ❌
```

runfile_2 must set `CFG.run_segmentation = False` and `CFG.mask_dir = './working/masks'` to reuse runfile_0's masks.

---

## Part 3: Complete Solution Scaffold

```python
# ── ranzcr_clip.py ───────────────────────────────────────────────────────────
import os, gc, time, random, warnings
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import timm
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
class CFG:
    seed          = 42
    debug         = False
    device        = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Stage 1: Segmentation ──────────────────────────────────────────────
    run_segmentation = True          # set False if masks already exist (runfile_2 must set False)
    seg_encoder      = 'resnet50'    # valid smp encoder; do NOT use 'tf_efficientnet_b1_ns'
    seg_epochs       = 10            # 10 epochs sufficient for mask quality; DO NOT increase (CG-10)
    seg_batch_size   = 16

    # ── Stage 2: Classification ────────────────────────────────────────────
    model_name    = 'tf_efficientnet_b5_ns'
    pretrained    = True
    n_ch          = 5               # 3 for RGB-only; 5 to stack segmentation masks
    img_size      = 640
    num_classes   = 12              # 11 submission labels + 1 synthetic no_ETT
    # Paths
    data_dir      = './input'
    img_folder    = 'train'
    mask_dir      = './working/masks'
    train_csv     = './input/train.csv'
    test_csv      = './input/sample_submission.csv'
    sub_csv       = './input/sample_submission.csv'
    # Training
    n_fold        = 3               # DO NOT increase to 5 (CG-10)
    batch_size    = 8
    lr            = 3e-4
    min_lr        = 1e-6
    weight_decay  = 1e-6
    warmup_epo    = 1
    cosine_epo    = 19              # DO NOT increase (CG-10)
    epochs        = warmup_epo + cosine_epo   # 20 total
    num_workers   = 4
    use_amp       = True
    use_tta       = True
    use_rank_cal  = True
    loss_weights  = (1., 9.)
    target_cols   = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
        'Swan Ganz Catheter Present',
    ]

assert len(CFG.target_cols) == 11


def seed_everything(seed):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(CFG.seed)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Segmentation (UNet++)
# ══════════════════════════════════════════════════════════════════════════════

class SegDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df          = df.reset_index(drop=True)
        self.transform   = transform
        self.data_dir    = CFG.data_dir
        self.img_folder  = CFG.img_folder
        self.mask_label_dir = CFG.get('mask_label_dir', None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = cv2.imread(
            os.path.join(self.data_dir, self.img_folder, row.StudyInstanceUID + '.jpg')
        )[:, :, ::-1].astype(np.float32)

        if self.mask_label_dir is not None:
            mask_path = os.path.join(self.mask_label_dir, row.StudyInstanceUID + '.png')
            mask_raw  = cv2.imread(mask_path)
            if mask_raw is not None:
                mask = mask_raw[:, :, :2].astype(np.float32) / 255.
            else:
                mask = np.zeros((img.shape[0], img.shape[1], 2), np.float32)
        else:
            mask = np.zeros((img.shape[0], img.shape[1], 2), np.float32)

        if self.transform:
            res  = self.transform(image=img, mask=mask)
            img  = res['image'].transpose(2, 0, 1) / 255.
            mask = res['mask'].transpose(2, 0, 1)
        else:
            img  = cv2.resize(img, (CFG.img_size, CFG.img_size)).transpose(2, 0, 1) / 255.
            mask = cv2.resize(mask, (CFG.img_size, CFG.img_size)).transpose(2, 0, 1)

        return (torch.tensor(img, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32))


class SegmentationModel(nn.Module):
    def __init__(self, encoder_name=CFG.seg_encoder, pretrained=True):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3,
            classes=2,
        )

    def forward(self, x):
        return self.model(x)


_seg_bce = nn.BCEWithLogitsLoss()

def seg_criterion(pred, target):
    bce          = _seg_bce(pred, target)
    pred_sig     = pred.sigmoid()
    intersection = (pred_sig * target).sum(dim=(2, 3))
    dice = 1 - (2 * intersection + 1) / (
        pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1)
    return 0.5 * bce + 0.5 * dice.mean()


def get_seg_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(CFG.img_size, CFG.img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                               rotate_limit=15, border_mode=0, p=0.5),
        ])
    return A.Compose([A.Resize(CFG.img_size, CFG.img_size)])


def train_segmentation(train_df, cfg=CFG):
    os.makedirs(cfg.mask_dir, exist_ok=True)

    seg_ds     = SegDataset(train_df, transform=get_seg_transforms('train'))
    seg_loader = DataLoader(seg_ds, batch_size=cfg.seg_batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    seg_model = SegmentationModel().to(cfg.device)
    optimizer = Adam(seg_model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.seg_epochs, eta_min=1e-6)
    scaler    = GradScaler(enabled=cfg.use_amp)

    for epoch in range(1, cfg.seg_epochs + 1):
        seg_model.train()
        losses = []
        for imgs, masks in seg_loader:
            imgs, masks = imgs.to(cfg.device), masks.to(cfg.device)
            optimizer.zero_grad()
            try:
                with autocast(enabled=cfg.use_amp):
                    pred = seg_model(imgs)
                    loss = seg_criterion(pred, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                losses.append(loss.item())
            except Exception as e:
                print(f'  [SEG WARNING] step skipped: {e}')
        scheduler.step()
        print(f'  Seg Epoch {epoch}/{cfg.seg_epochs} | loss={np.mean(losses):.4f}')

    seg_model.eval()
    test_df  = pd.read_csv(cfg.test_csv)
    all_uids = pd.concat([train_df[['StudyInstanceUID']],
                          test_df[['StudyInstanceUID']]]).drop_duplicates()

    with torch.no_grad():
        for uid in all_uids['StudyInstanceUID']:
            img_path = os.path.join(cfg.data_dir, 'train', uid + '.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(cfg.data_dir, 'test', uid + '.jpg')
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img[:, :, ::-1].astype(np.float32),
                             (cfg.img_size, cfg.img_size)) / 255.
            x   = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).to(cfg.device)
            mask = seg_model(x).sigmoid().squeeze(0).cpu().numpy()
            mask_hw2  = (mask.transpose(1, 2, 0) * 255).astype(np.uint8)
            save_img  = np.concatenate(
                [mask_hw2, np.zeros((cfg.img_size, cfg.img_size, 1), np.uint8)], axis=2)
            cv2.imwrite(os.path.join(cfg.mask_dir, uid + '.png'), save_img)

    print(f'Masks saved to {cfg.mask_dir}')
    del seg_model; gc.collect(); torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Classification
# ══════════════════════════════════════════════════════════════════════════════

def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(CFG.img_size, CFG.img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=0.75),
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=0.75),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.0),
            ], p=0.75),
            A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40,
                                  val_shift_limit=0, p=0.75),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3,
                                rotate_limit=30, border_mode=0, p=0.75),
            A.CoarseDropout(max_holes=2,
                            max_height=int(CFG.img_size * 0.3),
                            max_width=int(CFG.img_size * 0.3), p=0.5),
        ])
    return A.Compose([A.Resize(CFG.img_size, CFG.img_size)])


class RANZCRDataset(Dataset):
    def __init__(self, df, mode='train', transform=None):
        self.df         = df.reset_index(drop=True)
        self.mode       = mode
        self.transform  = transform
        self.data_dir   = CFG.data_dir
        self.img_folder = CFG.img_folder
        self.mask_dir   = CFG.mask_dir
        self.n_ch       = CFG.n_ch
        ordered = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 'no_ETT',
                   'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged',
                   'NGT - Normal', 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                   'Swan Ganz Catheter Present']
        self.labels = df[ordered].values if mode != 'test' else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = os.path.join(self.data_dir, self.img_folder,
                                row.StudyInstanceUID + '.jpg')
        try:
            img = cv2.imread(img_path)[:, :, ::-1].astype(np.float32)
        except Exception:
            img = np.zeros((CFG.img_size, CFG.img_size, 3), dtype=np.float32)

        if self.mask_dir is not None and self.n_ch == 5:
            try:
                mask = cv2.imread(
                    os.path.join(self.mask_dir, row.StudyInstanceUID + '.png')
                ).astype(np.float32)[:, :, :2]
                res   = self.transform(image=img, mask=mask)
                image = res['image'].transpose(2, 0, 1) / 255.
                mask  = res['mask'].transpose(2, 0, 1) / 255.
                image = np.concatenate([image, mask], axis=0)
            except Exception:
                res   = self.transform(image=img)
                image = res['image'].transpose(2, 0, 1) / 255.
                image = np.concatenate(
                    [image, np.zeros((2, CFG.img_size, CFG.img_size), np.float32)], axis=0)
        else:
            res   = self.transform(image=img)
            image = res['image'].transpose(2, 0, 1) / 255.

        image = torch.tensor(image, dtype=torch.float32)
        if self.labels is not None:
            label = torch.tensor(self.labels[index], dtype=torch.float32)
            return image, label
        return image


class RANZCRModel(nn.Module):
    def __init__(self, enet_type=CFG.model_name, out_dim=CFG.num_classes,
                 n_ch=CFG.n_ch, pretrained=CFG.pretrained):
        super().__init__()
        self.enet = timm.create_model(enet_type, pretrained=pretrained)
        if n_ch != 3:
            # EfficientNet → conv_stem | NFNet (eca_nfnet_*) → stem.conv | ResNet → conv1
            if hasattr(self.enet, 'conv_stem'):
                first_conv = self.enet.conv_stem
            elif hasattr(self.enet, 'stem') and hasattr(self.enet.stem, 'conv'):
                first_conv = self.enet.stem.conv
            elif hasattr(self.enet, 'conv1'):
                first_conv = self.enet.conv1
            else:
                raise AttributeError(f"Cannot find first conv in {enet_type}")
            first_conv.weight = nn.Parameter(
                first_conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
        self.dropout = nn.Dropout(0.5)
        # head: classifier (EfficientNet/NFNet) or fc (ResNet)
        if hasattr(self.enet, 'classifier'):
            in_features = self.enet.classifier.in_features
            self.enet.classifier = nn.Identity()
        elif hasattr(self.enet, 'fc'):
            in_features = self.enet.fc.in_features
            self.enet.fc = nn.Identity()
        else:
            in_features = self.enet.num_features
        self.myfc = nn.Linear(in_features, out_dim)

    def forward(self, x):
        features = self.enet(x)
        return self.myfc(self.dropout(features))


_ce  = nn.CrossEntropyLoss()
_bce = nn.BCEWithLogitsLoss()

def criterion(logits, targets):
    lw = CFG.loss_weights
    targets_smooth = targets.clone()
    targets_smooth[:, 4:] = targets[:, 4:].clamp(0.05, 0.95)
    loss_ce  = _ce(logits[:, :4], targets[:, :4].argmax(dim=1)) * lw[0]
    loss_bce = _bce(logits[:, 4:], targets_smooth[:, 4:]) * lw[1]
    return (loss_ce + loss_bce) / sum(lw)

def activation_split(logits):
    probs = logits.clone()
    probs[:, :4] = logits[:, :4].softmax(dim=1)
    probs[:, 4:] = logits[:, 4:].sigmoid()
    return probs

@torch.no_grad()
def predict_with_tta(model, images):
    p0 = activation_split(model(images))
    p1 = activation_split(model(images.flip(-1)))
    p2 = activation_split(model(images.flip(-2)))
    p3 = activation_split(model(images.flip([-1, -2])))
    return (p0 + p1 + p2 + p3) / 4.0

def rank_calibrate(fold_outputs_list):
    ranked = [pd.DataFrame(arr).rank(pct=True).values for arr in fold_outputs_list]
    return np.mean(ranked, axis=0)


def warmup_lr(optimizer, epoch, warmup_epo, base_lr):
    if epoch <= warmup_epo:
        lr = base_lr * (epoch / warmup_epo)
        for pg in optimizer.param_groups:
            pg['lr'] = lr


def train_one_epoch(model, loader, optimizer, epoch, device):
    model.train()
    scaler = GradScaler(enabled=CFG.use_amp)
    losses = []
    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        try:
            if np.random.rand() < 0.5:
                lam = float(np.random.beta(0.4, 0.4))
                idx = torch.randperm(images.size(0), device=device)
                images = lam * images + (1 - lam) * images[idx]
                labels = lam * labels + (1 - lam) * labels[idx]
            with autocast(enabled=CFG.use_amp):
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            losses.append(loss.item())
        except Exception as e:
            print(f'  [WARNING] step {step} skipped: {e}')
            continue
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def valid_one_epoch(model, loader, device):
    model.eval()
    losses, all_probs, all_labels = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        try:
            logits = model(images)
            loss   = criterion(logits, labels)
            losses.append(loss.item())
            if CFG.use_tta:
                probs = predict_with_tta(model, images).cpu().numpy()
            else:
                probs = activation_split(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
        except Exception as e:
            print(f'  [WARNING] val step skipped: {e}')
            continue
    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return float(np.mean(losses)), all_probs, all_labels


def get_auc(labels, probs):
    cols = list(range(3)) + list(range(4, 12))
    aucs = []
    for c in cols:
        try:
            aucs.append(roc_auc_score(labels[:, c], probs[:, c]))
        except Exception:
            pass
    return float(np.mean(aucs))


def run_cv(train_df, cfg=CFG):
    no_ett = (train_df[['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal']]
              .values.max(1) == 0).astype(int)
    ett_idx = list(train_df.columns).index('ETT - Normal') + 1
    train_df.insert(ett_idx, 'no_ETT', no_ett)

    gkf    = GroupKFold(n_splits=cfg.n_fold)
    groups = train_df['PatientID'].values if 'PatientID' in train_df.columns \
             else train_df.index.values

    oof_probs  = np.zeros((len(train_df), 12))
    oof_labels = np.zeros((len(train_df), 12))

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups)):
        print(f'\n========== Fold {fold+1}/{cfg.n_fold} ==========')
        trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)

        trn_ds = RANZCRDataset(trn_df, mode='train', transform=get_transforms('train'))
        val_ds = RANZCRDataset(val_df, mode='valid', transform=get_transforms('valid'))
        trn_loader = DataLoader(trn_ds, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=True)

        model     = RANZCRModel().to(cfg.device)
        optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.cosine_epo, eta_min=cfg.min_lr)

        best_auc  = 0.0
        ckpt_path = f'{cfg.model_name}_fold{fold}_best.pth'

        for epoch in range(1, cfg.epochs + 1):
            if epoch <= cfg.warmup_epo:
                warmup_lr(optimizer, epoch, cfg.warmup_epo, cfg.lr)
            else:
                scheduler.step()

            t0       = time.time()
            trn_loss = train_one_epoch(model, trn_loader, optimizer, epoch, cfg.device)
            val_loss, val_probs, val_labels = valid_one_epoch(model, val_loader, cfg.device)
            val_auc  = get_auc(val_labels, val_probs)
            elapsed  = time.time() - t0

            print(f'  Epoch {epoch:02d}/{cfg.epochs} | '
                  f'lr={optimizer.param_groups[0]["lr"]:.2e} | '
                  f'trn={trn_loss:.4f} | val={val_loss:.4f} | '
                  f'AUC={val_auc:.4f} | {elapsed:.0f}s')

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), ckpt_path)
                print(f'  ✓ checkpoint saved (AUC={val_auc:.4f})')

        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.to(cfg.device)
        _, fold_probs, fold_labels = valid_one_epoch(model, val_loader, cfg.device)
        oof_probs[val_idx]  = fold_probs
        oof_labels[val_idx] = fold_labels

        del model; gc.collect(); torch.cuda.empty_cache()

    oof_auc = get_auc(oof_labels, oof_probs)
    print(f'\nOOF AUC: {oof_auc:.4f}')
    return oof_probs, oof_labels


@torch.no_grad()
def run_test_inference(test_df, cfg=CFG):
    test_ds     = RANZCRDataset(test_df, mode='test', transform=get_transforms('valid'))
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size * 2,
                             shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    model = RANZCRModel().to(cfg.device)
    model.eval()

    fold_preds = []
    for fold in range(cfg.n_fold):
        ckpt = f'{cfg.model_name}_fold{fold}_best.pth'
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        model.to(cfg.device)
        fold_probs = []
        for images in test_loader:
            images = images.to(cfg.device)
            if cfg.use_tta:
                probs = predict_with_tta(model, images)
            else:
                probs = activation_split(model(images))
            fold_probs.append(probs.cpu().numpy())
        fold_preds.append(np.concatenate(fold_probs, axis=0))

    if cfg.use_rank_cal:
        preds_avg = rank_calibrate(fold_preds)
    else:
        preds_avg = np.mean(fold_preds, axis=0)

    return np.delete(preds_avg, 3, axis=1)   # drop no_ETT col → [N, 11]


def main():
    seed_everything(CFG.seed)

    train_df = pd.read_csv(CFG.train_csv)
    sample_sub = pd.read_csv(CFG.sub_csv)
    test_df  = sample_sub[['StudyInstanceUID']].copy()

    if CFG.run_segmentation:
        print('=== Stage 1: Training segmentation model and generating masks ===')
        train_segmentation(train_df)
    else:
        print(f'=== Stage 1 skipped — using existing masks in {CFG.mask_dir} ===')

    if not os.path.isdir(CFG.mask_dir) or len(os.listdir(CFG.mask_dir)) == 0:
        print('[WARNING] mask_dir is empty — falling back to 3-channel mode')
        CFG.mask_dir = None
        CFG.n_ch     = 3

    print('\n=== Stage 2: Training classification model ===')
    oof_probs, oof_labels = run_cv(train_df)

    test_preds = run_test_inference(test_df)

    sub = pd.read_csv(CFG.sub_csv)
    for i, col in enumerate(CFG.target_cols):
        sub[col] = test_preds[:, i]
    sub.to_csv('./submission/submission.csv', index=False)
    print('Saved submission.csv'); print(sub.head())


if __name__ == '__main__':
    os.makedirs('./submission', exist_ok=True)
    os.makedirs('./working',    exist_ok=True)
    main()
```

---

## Part 4: Output Specification and Trigger

Generate **3 runfiles + 1 ensemble script**. Each runfile is the scaffold above with only CFG changes:

| File | `model_name` | `img_size` | `batch_size` | `run_segmentation` |
|------|-------------|------------|--------------|-------------------|
| `runfile_0.py` | `tf_efficientnet_b5_ns` | 640 | 8 | `True` |
| `runfile_1.py` | `resnet200d` | 640 | 8 | `True` |
| `runfile_2.py` | `eca_nfnet_l0` | 512 | 16 | `False` + `mask_dir='./working/masks'` |

**`ensemble.py`** (averages rank-calibrated predictions from all three runfiles):
```python
import pandas as pd, numpy as np

paths = ['./submission_rf0/submission.csv',
         './submission_rf1/submission.csv',
         './submission_rf2/submission.csv']
subs = [pd.read_csv(p) for p in paths]
uid_col = 'StudyInstanceUID'
target_cols = [c for c in subs[0].columns if c != uid_col]

ranked = [sub[target_cols].rank(pct=True) for sub in subs]
avg = np.mean([r.values for r in ranked], axis=0)

final = subs[0][[uid_col]].copy()
for i, col in enumerate(target_cols):
    final[col] = avg[:, i]
final.to_csv('./submission/submission.csv', index=False)
print('Ensemble submission saved.')
```

**Now produce the four files. Copy the scaffold verbatim; change only the CFG values listed above. Do NOT generate a simplified baseline.**
