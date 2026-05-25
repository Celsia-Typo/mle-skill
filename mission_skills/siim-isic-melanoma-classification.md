# SIIM-ISIC Melanoma Classification Task SCoT Skill

## 1. Instructions

You are a medical image deep learning expert specializing in the SIIM-ISIC melanoma binary classification task (malignant vs. benign).

**Goals & Constraints:**
* Input: Dermoscopy images and optional patient metadata (age, sex, anatomical site).
* Output: Malignant probability score (0~1) for each image. The evaluation metric is AUC-ROC.
* Target: Attain a Medal-winning score. This REQUIRES moving beyond single-model setups.
* Constraint 1: DO NOT rely solely on a single backbone. To reach the actual medal zone, you MUST orchestrate a Multi-Model Ensemble (e.g., merging predictions from B4, B5, B6 versions at different resolutions).
* Constraint 2: External data (ISIC 2019/2018) is strictly NOT available. You must use pseudo-labeling as the primary substitute.
* Constraint 3 (EXTREME I/O OPTIMIZATION): Do NOT read millions of small JPEGs directly from standard disk during training. You MUST use a RAM disk. Implement logic that checks if the dataset exists in `/dev/shm` (or equivalent Linux memory disk). If not, copy the JPEGs to `/dev/shm` ONCE at the start of the script, and configure Datasets to point there.
* Constraint 4 (FAST IMAGE LOADING): Do NOT use `PIL.Image.open()` for reading images. You MUST use OpenCV (`cv2.imread(filepath)` and `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`). To prevent multi-processing freezes, you MUST explicitly set `cv2.setNumThreads(0)` at the top level of your script.
* Constraint 5 (DATALOADER TUNING): PyTorch `DataLoader` must use `num_workers=8` (or higher), `pin_memory=True`, `persistent_workers=True`, and `prefetch_factor=2` to ensure the GPUs are never starved.
* Constraint 6 (FOLD SPLIT — CRITICAL): You MUST use `StratifiedGroupKFold(n_splits=5)` and pass `groups=df["patient_id"]` to `.split()`. Do NOT use plain `StratifiedKFold`. Without grouping by patient, the same patient's lesions leak across train/val, causing inflated AUC and wrong checkpoint selection. Fill missing `patient_id` with the scalar string `"unknown"` before splitting.
* Constraint 7 (TRAINING EPOCHS): Set `NUM_EPOCHS = 20` for B4 models and `NUM_EPOCHS = 15` for B5 models. Training for only 10 epochs underfits; the model has not converged and is leaving 0.003–0.006 AUC on the table.
* Constraint 8 (CLASS IMBALANCE SAMPLER): The positive rate is ~1.76 %. You MUST use `WeightedRandomSampler` instead of `shuffle=True` on the train DataLoader. Without it, many batches contain zero positive samples and produce zero gradients. Compute per-sample weights as `w = n_total / (2 * n_class)` and pass to `WeightedRandomSampler`.
* Constraint 9 (NEAR-DUPLICATE REMOVAL): Before splitting folds, you MUST remove near-duplicate images using perceptual hashing (difference hash). There are ~1800 near-duplicate frames of the same lesion; if they straddle train/val they cause data leakage and over-optimistic AUC.
* Constraint 10 (TTA + HAIR AUGMENTATION): Training augmentation MUST include `HairAugmentation` (synthetic Bezier-curve hair strokes implemented as an `albumentations.ImageOnlyTransform`). TTA at inference MUST cover at least 5 views: original, horizontal flip, vertical flip, 90-degree rotation, and 270-degree rotation. Using only 3 views leaves 0.002–0.005 AUC unrealised.
* Constraint 11 (PSEUDO-LABEL VAL FILTER — CRITICAL): When building the round-2 dataset by concatenating `train_df` and `pseudo_df` (test images), you MUST mark pseudo rows with `is_pseudo=True`. The validation fold for round 2 MUST be filtered to contain only `is_pseudo==False` rows. If pseudo images enter the val fold, they are looked up in `TRAIN_IMG_DIR` (not `TEST_IMG_DIR`) causing `FileNotFoundError`; and val AUC becomes artificially high, causing the wrong checkpoint to be saved. Additionally, store absolute `img_path` per row in both DataFrames before concatenating, and have `MelanomaDataset.__getitem__` read `row["img_path"]` directly instead of `os.path.join(self.img_dir, ...)`.
* Constraint 12 (DDP FOR runfile_1.py): `runfile_1.py` must use `DistributedDataParallel` (DDP) instead of `nn.DataParallel`. Set `CUDA_VISIBLE_DEVICES=2,3` before any CUDA import, initialize with `dist.init_process_group(backend="nccl", init_method="env://")`, and wrap the model with `nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])`. Use `DistributedSampler` for all DataLoaders and call `train_sampler.set_epoch(epoch)` at the start of every epoch. Only the main process (`is_main_process`) saves checkpoints; unwrap via `model.module` before saving. Cast predictions to `float32` explicitly before loss computation (`criterion(preds.float(), labels.float())`) and use `non_blocking=True` on all `.to(device)` calls. Launch with `torchrun --nproc_per_node=2 runfile_1.py`.

**Reasoning Requirements:**
Before writing the code, please write the problem-solving process using SCoT (Structured Chain-of-Thought, similar to pseudocode). Explicitly define the IO structure, comprehensively apply sequential, branch, and loop structures, and then output the final code. Convert SCoT into single-line comments embedded in the code block. Pay specific attention to the memory copying logic and DataLoader construction in your SCoT.

## 2. Demonstration Examples

### Example 1: Building an EfficientNet Model Fusing Metadata

**[Requirement]**
```python
def build_model(enet_type: str, out_dim: int,
                n_meta_features: int = 0, n_meta_dim: list = None,
                pretrained: bool = True) -> nn.Module:
    """Build Effnet model with an implicit multi-path dropout ensemble head."""
```

**[SCoT]**
```python
# IO Definition
# Input: enet_type(str), out_dim(int), n_meta_features(int)
# Output: nn.Module

# Sequential Structure - Load EfficientNet backbone
# extract backbone features
# Branch Structure - Whether to concatenate metadata features
# if n_meta_features > 0:
#     append metadata dense network layers
# Loop Structure - Define multi-path Dropout classification head (average of 5 paths)
# process forward pass iterating over 5 dropouts and accumulate
```

**[Code]**
```python
import torch
import torch.nn as nn
import geffnet

class Effnet_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=None):
        super(Effnet_Melanoma, self).__init__()
        # Sequential Structure - Load EfficientNet backbone
        self.enet = geffnet.create_model(enet_type, pretrained=True)
        in_ch = self.enet.classifier.in_features
        self.enet.classifier = nn.Identity()

        # Branch Structure - Whether to concatenate metadata features
        self.n_meta_features = n_meta_features
        if self.n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                nn.SiLU(),
                nn.Dropout(p=0.2),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                nn.SiLU(),
            )
            in_ch += n_meta_dim[1]

        # Loop Structure - implicit ensemble inside head
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.myfc = nn.Linear(in_ch, out_dim)

    def forward(self, x, x_meta=None):
        x = self.enet(x)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)

        # Loop to average 5 dropout paths
        out = sum([self.myfc(drop(x)) for drop in self.dropouts]) / len(self.dropouts)
        return out
```

### Example 2: Rank-Based Multi-Model Ensemble

**[Requirement]**
```python
def rank_average_ensemble(sub_files: list, weights: list = None,
                          output_path: str = "ensemble_submission.csv") -> pd.DataFrame:
    """Ensemble multiple submission CSVs using rank averaging to fix model scale differences."""
```

**[SCoT]**
```python
# IO Definition
# Input: sub_files(list), weights(list)
# Output: DataFrame

# Branch Structure - Default to equal weights if none
# if not weights: weights = [1.0 / len(sub_files)] * len(sub_files)

# Sequential - initialize base DataFrame
# base_df = read first sub_file

# Loop Structure - Rank Average Transformation
# for each file and weight:
#    read df
#    rank target column (pct=True)
#    accumulate weighted rank
```

**[Code]**
```python
import pandas as pd

def rank_average_ensemble(sub_files: list, weights: list = None,
                          output_path: str = "ensemble_submission.csv") -> pd.DataFrame:
    # Branch Structure - Default to equal weights if none
    if weights is None:
        weights = [1.0 / len(sub_files)] * len(sub_files)

    df_out = pd.read_csv(sub_files[0])
    df_out['target'] = 0.0

    # Loop Structure - Rank Average Transformation
    for file, w in zip(sub_files, weights):
        df_temp = pd.read_csv(file)
        # Apply rank percentage transformation
        df_temp['target'] = df_temp['target'].rank(pct=True)
        df_out['target'] += df_temp['target'] * w

    df_out.to_csv(output_path, index=False)
    return df_out
```

### Example 3: Extreme I/O Acceleration with RAM Disk & OpenCV

**[Requirement]**
```python
def setup_dataset_in_ram(src_dir: str, dest_dir: str = "/dev/shm/dataset") -> str:
    """Copies dataset to RAM disk for ultra-fast training iterations."""
```

**[SCoT]**
```python
# IO Definition
# Input: src_dir (str), dest_dir (str)
# Output: working_directory path (str)

# Sequential Structure - Apply multiprocessing configuration for deep learning
# set cv2 to use 0 threads to avoid deadlocking with PyTorch Dataloaders

# Branch Structure - Check if data already exists in RAM
# if data is not already in dest_dir:
#     Sequential - Execute recursive copy
#     shutil.copytree(src_dir, dest_dir)
# return dest_dir path for DataLoader use
```

**[Code]**
```python
import os
import shutil
import cv2

# CRUCIAL: Prevent OpenCV from crashing dataloader workers
cv2.setNumThreads(0)

def setup_dataset_in_ram(src_dir: str, dest_dir: str = "/dev/shm/dataset") -> str:
    # Branch Structure - Check if data already exists in RAM
    if not os.path.exists(dest_dir):
        print(f"Transferring {src_dir} to RAM Disk ({dest_dir})...")
        # Sequential - Execute recursive copy
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
        print("Transfer complete.")
    else:
        print("Data already cached in RAM disk.")

    return dest_dir
```

### Example 4: Near-Duplicate Removal + StratifiedGroupKFold Split

**[Requirement]**
```python
def deduplicate_and_split(
    train_df: pd.DataFrame,
    img_dir: str,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple:
    """Remove near-duplicate images via difference hash, then build a
    StratifiedGroupKFold splitter grouped by patient_id to prevent
    patient-level data leakage across train and val folds."""
```

**[SCoT]**
```python
# IO Definition
# Input: train_df(DataFrame), img_dir(str), n_folds(int), seed(int)
# Output: (deduplicated train_df, sgkf splitter, groups array)

# Sequential - step 1: compute difference hash for every training image
# Sequential - step 2: drop rows whose hash was seen before (keep first occurrence)
# Branch: if "patient_id" column missing, create it filled with "unknown"
# Sequential - step 3: fillna patient_id with scalar "unknown"
# Sequential - step 4: build StratifiedGroupKFold(n_splits=n_folds, shuffle=True)
# Sequential - step 5: groups = train_df["patient_id"].values
# Loop: fold loop — always pass groups= to sgkf.split()
```

**[Code]**
```python
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def _dhash(img: np.ndarray, hash_size: int = 16) -> int:
    # Sequential: compute difference hash (no extra package needed)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (hash_size + 1, hash_size))
    diff  = small[:, 1:] > small[:, :-1]
    return sum(2 ** i for i, bit in enumerate(diff.flatten()) if bit)

def deduplicate_and_split(train_df, img_dir, n_folds=5, seed=42):
    # Sequential: hash every image, drop duplicates before splitting
    n_orig = len(train_df)
    seen, drop_ids = {}, set()
    for img_name in train_df["image_name"]:
        path = os.path.join(img_dir, f"{img_name}.jpg")
        img  = cv2.imread(path)
        if img is None:
            continue
        h = _dhash(img)
        if h in seen:
            drop_ids.add(img_name)
        else:
            seen[h] = img_name
    train_df = train_df[~train_df["image_name"].isin(drop_ids)].reset_index(drop=True)
    print(f"[dedup] {n_orig} -> {len(train_df)} (removed {len(drop_ids)} near-duplicates)")

    # Branch: guard missing patient_id column
    if "patient_id" not in train_df.columns:
        train_df["patient_id"] = "unknown"
    train_df["patient_id"] = train_df["patient_id"].fillna("unknown")

    sgkf   = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    groups = train_df["patient_id"].values

    # Loop: fold loop — groups= is mandatory; never omit it
    for fold, (trn_idx, val_idx) in enumerate(
            sgkf.split(train_df, train_df["target"], groups)):
        ...

    return train_df, sgkf, groups
```

### Example 5: WeightedRandomSampler for Class Imbalance

**[Requirement]**
```python
def build_weighted_train_loader(
    dataset: Dataset,
    targets: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Build a train DataLoader with WeightedRandomSampler to compensate for
    the ~1.76% positive rate; replaces shuffle=True entirely."""
```

**[SCoT]**
```python
# IO Definition
# Input: dataset(Dataset), targets(np.ndarray of 0/1), batch_size(int), num_workers(int)
# Output: DataLoader with WeightedRandomSampler

# Sequential - step 1: count positives and negatives
# Sequential - step 2: compute inverse-frequency weight per class
#   w_pos = n_total / (2 * n_pos);  w_neg = n_total / (2 * n_neg)
# Sequential - step 3: assign per-sample weight array
# Sequential - step 4: build WeightedRandomSampler(weights, num_samples, replacement=True)
# Sequential - step 5: pass sampler= to DataLoader; do NOT also set shuffle=True
```

**[Code]**
```python
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import torch

def build_weighted_train_loader(dataset, targets, batch_size, num_workers):
    # Sequential: compute inverse-frequency per-sample weights
    targets  = targets.astype(int)
    n_pos    = int(targets.sum())
    n_neg    = int(len(targets) - n_pos)
    n_total  = len(targets)
    w_pos    = n_total / (2.0 * max(n_pos, 1))
    w_neg    = n_total / (2.0 * max(n_neg, 1))
    sample_w = torch.from_numpy(
        np.where(targets == 1, w_pos, w_neg).astype(np.float32))

    # Sequential: build sampler and DataLoader
    sampler = WeightedRandomSampler(
        weights=sample_w, num_samples=n_total, replacement=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,           # replaces shuffle=True
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
```

### Example 6: HairAugmentation + 5-View Rotation TTA

**[Requirement]**
```python
def get_augmentation_pipeline(img_size: int, mode: str = "train"):
    """Return an Albumentations pipeline for the given mode.
    mode='train': HairAugmentation + full stochastic augmentations.
    mode='tta':   list of 5 deterministic transforms (orig, hflip, vflip, rot90, rot270)."""
```

**[SCoT]**
```python
# IO Definition
# Input: img_size(int), mode(str: "train" | "tta")
# Output (train): A.Compose pipeline including HairAugmentation
# Output (tta):   list of 5 A.Compose pipelines, one per TTA view

# Branch Structure - mode == "train":
#   Sequential: RandomResizedCrop, flips, rotations, ColorJitter
#   Sequential: HairAugmentation (albumentations.ImageOnlyTransform subclass)
#   Sequential: noise/blur, CoarseDropout, Normalize, ToTensorV2

# Branch Structure - mode == "tta":
#   Sequential: build base norm ops list [Normalize, ToTensorV2]
#   Loop: for each of 5 views (orig, hflip, vflip, rot90, rot270)
#         compose [Resize + view-specific op + norm]

# HairAugmentation internal loop:
#   Loop: for _ in range(num_hairs): draw one quadratic Bezier curve
#   Loop: for i in range(n_pts+1):   interpolate and draw line segments
```

**[Code]**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2, random as rnd, numpy as np

class HairAugmentation(A.ImageOnlyTransform):
    """Overlay synthetic Bezier-curve hair strokes for dermoscopy robustness."""
    def __init__(self, num_hairs=5, thickness_range=(1, 3), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.num_hairs = num_hairs
        self.thickness_range = thickness_range

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.copy()
        h, w = img.shape[:2]
        # Loop: draw each hair as a quadratic Bezier curve
        for _ in range(self.num_hairs):
            x1, y1 = rnd.randint(0, w-1), rnd.randint(0, h-1)
            x2, y2 = rnd.randint(0, w-1), rnd.randint(0, h-1)
            cx = rnd.randint(min(x1, x2), max(x1, x2) + 1)
            cy = rnd.randint(min(y1, y2), max(y1, y2) + 1)
            color = (rnd.randint(5, 30), rnd.randint(5, 20), rnd.randint(5, 20))
            thickness = rnd.randint(*self.thickness_range)
            n_pts, prev = 100, None
            # Loop: Bezier interpolation -> line segments
            for i in range(n_pts + 1):
                t  = i / n_pts
                px = int((1-t)**2*x1 + 2*(1-t)*t*cx + t**2*x2)
                py = int((1-t)**2*y1 + 2*(1-t)*t*cy + t**2*y2)
                if prev:
                    cv2.line(img, prev, (px, py), color, thickness, lineType=cv2.LINE_AA)
                prev = (px, py)
        return img

    def get_transform_init_args_names(self):
        return ("num_hairs", "thickness_range")

def get_augmentation_pipeline(img_size, mode="train"):
    _NORM = [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]

    # Branch: training pipeline
    if mode == "train":
        return A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            HairAugmentation(num_hairs=5, p=0.5),
            A.OneOf([A.GaussNoise(), A.GaussianBlur(), A.MotionBlur()], p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        ] + _NORM)

    # Branch: TTA — 5 deterministic views
    return [
        A.Compose([A.Resize(img_size, img_size)] + _NORM),                                   # view 0: original
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0)] + _NORM),          # view 1: hflip
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.0)] + _NORM),            # view 2: vflip
        A.Compose([A.Resize(img_size, img_size), A.Rotate(limit=(90, 90), p=1.0)] + _NORM),  # view 3: rot90
        A.Compose([A.Resize(img_size, img_size), A.Rotate(limit=(-90,-90), p=1.0)] + _NORM), # view 4: rot270
    ]

# Loop: TTA inference over 5 views, average sigmoid probabilities
@torch.no_grad()
def predict_with_tta(model, df, meta_feats, img_size, device):
    model.eval()
    all_view_preds = []
    for tfm in get_augmentation_pipeline(img_size, mode="tta"):  # Loop: 5 views
        ds     = MelanomaDataset(df, meta_feats, transforms=tfm, is_test=True)
        loader = DataLoader(ds, batch_size=BATCH_SIZE*2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True, prefetch_factor=2)
        preds  = []
        for imgs, metas in loader:                                # Loop: batches
            imgs, metas = imgs.to(device), metas.to(device)
            with torch.cuda.amp.autocast():
                preds.append(torch.sigmoid(model(imgs, metas)).cpu().numpy())
        all_view_preds.append(np.concatenate(preds))
    return np.mean(all_view_preds, axis=0)
```

### Example 7: Pseudo-Labeling Round 2 with Val Contamination Fix

**[Requirement]**
```python
def build_pseudo_round2(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_preds: np.ndarray,
    TRAIN_IMG_DIR: str,
    TEST_IMG_DIR: str,
) -> pd.DataFrame:
    """Stamp absolute img_path per row, threshold round-1 predictions into
    pseudo-labels, tag them with is_pseudo=True and unique patient_id, then
    concatenate with the original training set for round-2 retraining.
    The caller must filter val folds to is_pseudo==False before building DataLoaders."""
```

**[SCoT]**
```python
# IO Definition
# Input: train_df (original), test_df (with img_path to TEST_IMG_DIR),
#        test_preds (round-1 np.ndarray), TRAIN_IMG_DIR(str), TEST_IMG_DIR(str)
# Output: df_aug (DataFrame) — combined train + pseudo rows

# Sequential - step 1: stamp img_path on both DataFrames right after CSV load
#   train_df["img_path"] = TRAIN_IMG_DIR / image_name + ".jpg"
#   test_df["img_path"]  = TEST_IMG_DIR  / image_name + ".jpg"
# Note: MelanomaDataset.__getitem__ must read row["img_path"] directly

# Sequential - step 2: threshold round-1 preds
# Branch: if no positives at 0.95 -> lower to 0.90
# Branch: if no negatives at 0.05 -> raise to 0.10

# Sequential - step 3: tag pseudo rows
#   df_pseudo["is_pseudo"]  = True
#   df_pseudo["patient_id"] = "pseudo_" + image_name  (unique -> stays in train fold)
#   train_df["is_pseudo"]   = False

# Sequential - step 4: align columns and concat
#   keep_cols = intersection of train_df.columns and df_pseudo.columns
#   df_aug = pd.concat([train_df, df_pseudo[keep_cols]])

# Loop: for each fold in sgkf.split(df_aug, df_aug["target"], groups_aug)
#   Sequential - build trn_df from trn_idx (may include pseudo rows)
#   Branch: filter val_df -> keep only is_pseudo==False rows
#   Branch: if filtered val is empty -> skip fold with warning
#   Sequential: MelanomaDataset reads row["img_path"] directly (no img_dir arg needed)
```

**[Code]**
```python
import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

# Sequential - step 1: stamp img_path BEFORE any concat (right after CSV load)
train_df["img_path"] = train_df["image_name"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, x + ".jpg"))
test_df["img_path"]  = test_df["image_name"].apply(
    lambda x: os.path.join(TEST_IMG_DIR, x + ".jpg"))

# MelanomaDataset reads row["img_path"] directly — no self.img_dir needed
class MelanomaDataset(Dataset):
    def __init__(self, df, meta_feats, transforms=None, is_test=False):
        self.df         = df.reset_index(drop=True)
        self.meta_feats = meta_feats
        self.transforms = transforms
        self.is_test    = is_test

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = row["img_path"]             # use pre-stored absolute path
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        meta = torch.tensor(row[self.meta_feats].values.astype(np.float32))
        if self.is_test:
            return img, meta
        return img, meta, torch.tensor(row["target"], dtype=torch.float32)

# Sequential - step 2: threshold round-1 predictions
pos_mask = test_preds > 0.95
neg_mask = test_preds < 0.05
if pos_mask.sum() == 0:         # Branch: fallback threshold
    pos_mask = test_preds > 0.90
if neg_mask.sum() == 0:
    neg_mask = test_preds < 0.10

df_pseudo_pos = test_df[pos_mask].copy(); df_pseudo_pos["target"] = 1
df_pseudo_neg = test_df[neg_mask].copy(); df_pseudo_neg["target"] = 0
df_pseudo = pd.concat([df_pseudo_pos, df_pseudo_neg], ignore_index=True)

# Sequential - step 3: tag pseudo rows
df_pseudo["is_pseudo"]  = True
df_pseudo["patient_id"] = "pseudo_" + df_pseudo["image_name"]
# img_path already set from test_df — do NOT overwrite

train_df["is_pseudo"] = False
keep_cols = [c for c in train_df.columns if c in df_pseudo.columns]
df_aug    = pd.concat([train_df, df_pseudo[keep_cols]], ignore_index=True)

# Loop: round-2 fold loop
groups_aug = df_aug["patient_id"].values
for fold, (trn_idx, val_idx) in enumerate(
        sgkf.split(df_aug, df_aug["target"], groups_aug)):

    trn_df = df_aug.iloc[trn_idx].reset_index(drop=True)

    # Branch: val fold must only contain original annotated images
    val_df = df_aug.iloc[val_idx]
    val_df = val_df[val_df["is_pseudo"] == False].reset_index(drop=True)
    if len(val_df) == 0:
        print(f"  [WARN] Fold {fold+1} val empty after pseudo filter — skipping")
        continue

    trn_dataset = MelanomaDataset(trn_df, meta_feats, transforms=get_augmentation_pipeline(IMG_SIZE, "train"))
    val_dataset = MelanomaDataset(val_df, meta_feats, transforms=get_augmentation_pipeline(IMG_SIZE, "val"))
    ...
```

### Example 8: DDP Initialization + Pipeline Optimization (runfile_1.py)

**[Requirement]**
```python
def init_ddp_training(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model: nn.Module,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple:
    """Initialize DistributedDataParallel (DDP) process group for runfile_1.py
    (GPU 2, 3), build DDP-aware DataLoaders with persistent workers and pinned
    memory, and wrap the model. Returns (model, train_loader, val_loader, device,
    is_main_process)."""
```

**[SCoT]**
```python
# IO Definition
# Input: datasets, model, batch_size, num_workers, seed
# Output: (ddp_model, train_loader, val_loader, device, is_main_process)

# Sequential - step 1: set CUDA_VISIBLE_DEVICES="2,3" before any CUDA import
# Branch: if RANK/WORLD_SIZE env vars present -> torchrun launch
#   dist.init_process_group(backend="nccl", init_method="env://")
#   local_rank = int(os.environ["LOCAL_RANK"])
#   is_main_process = (RANK == 0)
# Branch: else -> single-process fallback
#   dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:23456",
#                           rank=0, world_size=1)
#   local_rank = 0; is_main_process = True
# Sequential - step 2: torch.cuda.set_device(local_rank)

# Sequential - step 3: build DistributedSampler for each DataLoader
# Sequential - step 4: build DataLoaders with pin_memory + persistent_workers

# Sequential - step 5: model.to(device)
# Sequential - step 6: wrap with nn.parallel.DistributedDataParallel

# Loop: training loop
#   Sequential: train_sampler.set_epoch(epoch)  (mandatory for DDP shuffle)
#   Sequential: train_one_epoch(...)
# Branch: checkpoint save — only is_main_process saves; unwrap model.module first

# Sequential - dtype fix in train_one_epoch and validate:
#   images/metas/labels.to(device, non_blocking=True)
#   criterion(preds.float(), labels.float())  (explicit float32 to avoid AMP sync cost)
```

**[Code]**
```python
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# Sequential - step 1: GPU binding before any CUDA import
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import cv2
cv2.setNumThreads(0)

# Branch: torchrun launch vs. single-process fallback
if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank      = int(os.environ["LOCAL_RANK"])
    is_main_process = (int(os.environ["RANK"]) == 0)
else:
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:23456",
                             rank=0, world_size=1)
    local_rank      = 0
    is_main_process = True

# Sequential - step 2: bind current process to its GPU
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# Sequential - step 3 & 4: DDP-aware DataLoaders
train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=SEED)
val_sampler   = DistributedSampler(val_dataset,   shuffle=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    prefetch_factor=2, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE * 2, sampler=val_sampler,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    prefetch_factor=2,
)

# Sequential - step 5 & 6: move model and wrap with DDP
model = model.to(device)
model = nn.parallel.DistributedDataParallel(
    model, device_ids=[local_rank], output_device=local_rank)

# Loop: training loop — set_epoch is mandatory for DDP shuffle
for epoch in range(1, NUM_EPOCHS + 1):
    train_sampler.set_epoch(epoch)
    train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
    _, val_auc, _ = validate(model, val_loader, criterion, device)

    # Branch: only main process saves checkpoints
    if is_main_process and val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.module.state_dict(), best_model_path)  # unwrap before saving

# dtype-aligned train_one_epoch (explicit float32 cast eliminates AMP hidden sync)
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for images, metas, labels in loader:
        images = images.to(device, non_blocking=True)  # async copy via pin_memory
        metas  = metas.to(device,  non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            preds = model(images, metas)
            loss  = criterion(preds.float(), labels.float())  # explicit float32 cast
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for images, metas, labels in loader:
            images = images.to(device, non_blocking=True)
            metas  = metas.to(device,  non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                preds = model(images, metas)
                loss  = criterion(preds.float(), labels.float())
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(preds.float()).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Note: in DDP each rank sees only its own shard; for exact global AUC
    # gather all_preds/all_labels across ranks with dist.all_gather.
    auc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), auc, np.array(all_preds)
```

## 3. Testing Requirement

**[Requirement]**
Please implement the orchestrator script to process an advanced ensemble pipeline using models with multiple resolutions and architectures (e.g., `efficientnet_b4_ns` at 384px and `efficientnet_b5_ns` at 512px). Address the drawbacks of single-model execution by gathering predictions from these multiple variants, running test-time augmentation (TTA), and finally executing `rank_average_ensemble` to produce a robust medal-grade submission.

**GPU Allocation Constraint:**
* `runfile_0.py` must occupy **GPU 0** (set `CUDA_VISIBLE_DEVICES=0` at the top of the script).
* `runfile_1.py` must occupy **GPU 1** (set `CUDA_VISIBLE_DEVICES=1` at the top of the script).
* Each runfile should set its visible devices via `os.environ["CUDA_VISIBLE_DEVICES"]` before any torch/model initialization, so that the two scripts can be launched in parallel without GPU conflicts.

Let's think step by step. Write your code here:
```python
