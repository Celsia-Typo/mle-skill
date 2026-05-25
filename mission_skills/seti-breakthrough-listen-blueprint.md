# Blueprint: seti-breakthrough-listen.ipynb

## Overview

This notebook is a **PyTorch training pipeline** for the [SETI Breakthrough Listen](https://www.kaggle.com/c/seti-breakthrough-listen) Kaggle competition. The task is to classify radio signal spectrograms as either containing an anomalous (potentially artificial) signal or consisting of background noise only. The model uses a pretrained **NFNet-L0** architecture via `timm`, trained with 4-fold stratified cross-validation.

---

## Purpose

Train a binary image classifier on cadence snippets of radio frequency spectrograms to detect narrowband signals of interest, optimised for ROC-AUC.

---

## Dependencies

| Library | Role |
|---|---|
| `numpy`, `pandas` | Data handling |
| `PIL`, `cv2` | Image utilities |
| `torch`, `torch.nn` | Model and training |
| `timm` | Pretrained model library (NFNet-L0) |
| `albumentations` | Image augmentation |
| `sklearn` | StratifiedKFold, ROC-AUC metric |
| `torch.cuda.amp` | Mixed-precision training (optional) |
| `matplotlib`, `seaborn` | EDA visualization |

---

## Competition Details

| Field | Value |
|---|---|
| Task | Binary classification |
| Input | `.npy` arrays of shape `(6, 273, 256)` — 6 ON/OFF cadence images |
| Target | `1` = signal present, `0` = background only |
| Evaluation | ROC-AUC |
| Data | `train_labels.csv`, `sample_submission.csv`, `train/`, `test/` |

---

## Data Format

Each sample is a NumPy `.npy` file of shape `(6, 273, 256)`:
- 6 cadence images (alternating ON/OFF source observations)
- Stacked vertically → `(1638, 256)` after `np.vstack().transpose()`
- Treated as a **single-channel grayscale image** for the CNN

File paths follow the pattern: `{split}/{id[0]}/{id}.npy` (first character of ID used as subdirectory).

---

## Configuration (`CFG` class)

| Parameter | Value | Description |
|---|---|---|
| `model_name` | `'nfnet_l0'` | timm architecture |
| `size` | 224 | Image resize target |
| `epochs` | 6 | Training epochs |
| `scheduler` | `'CosineAnnealingLR'` | LR schedule |
| `T_max` | 6 | Cosine annealing period |
| `lr` | 1e-4 | Base learning rate |
| `min_lr` | 1e-6 | Minimum LR |
| `batch_size` | 64 | Training batch size |
| `weight_decay` | 1e-6 | Adam weight decay |
| `max_grad_norm` | 1000 | Gradient clipping |
| `n_fold` | 4 | Number of CV folds |
| `seed` | 42 | Global seed |
| `target_size` | 1 | Output dimension (binary) |
| `apex` | False | AMP mixed precision toggle |
| `debug` | False | Quick debug mode (1000 samples, 1 epoch) |

---

## Pipeline

### 1. EDA
```python
image = np.load(train.loc[i, 'file_path'])  # (6, 273, 256)
image = np.vstack(image).transpose((1, 0))   # → (256, 1638)
plt.imshow(image)
```
Displays first 10 training samples as stacked cadence spectrograms. Target distribution histogram confirms class imbalance.

### 2. Reproducibility Setup
Seeds `random`, `numpy`, `torch`, and `torch.cuda` uniformly. Enables `cudnn.deterministic`.

### 3. Stratified K-Fold Split
```python
Fold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
```
Assigns fold index to each row while preserving the positive/negative ratio across folds.

### 4. Dataset (`TrainDataset`)
```python
image = np.load(file_path)           # (6, 273, 256)
image = image.astype(np.float32)
image = np.vstack(image).transpose((1, 0))   # single 2D image
# Optionally apply Albumentations transform
label = torch.tensor(self.labels[idx]).float()
```

### 5. Transforms (`get_transforms`)
- **Train & Valid**: `Resize(224, 224)` → `ToTensorV2()`
- No augmentation beyond resize in this baseline (augmentation is a key improvement area).

### 6. Model (`CustomModel`)
```python
self.model = timm.create_model('nfnet_l0', pretrained=True, in_chans=1)
self.model.head.fc = nn.Linear(self.n_features, 1)  # binary output
```
NFNet-L0 is a Normalizer-Free ResNet variant; `in_chans=1` handles the single-channel input.

### 7. Training Loop (`train_fn`)
- Loss: `nn.BCEWithLogitsLoss`
- Optimizer: `Adam(lr=1e-4, weight_decay=1e-6)`
- Scheduler: `CosineAnnealingLR`
- Gradient clipping: `clip_grad_norm_(..., 1000)`
- Optional AMP (Automatic Mixed Precision) via `GradScaler`
- Logs every `print_freq=100` steps: loss, gradient norm, elapsed/remaining time

### 8. Validation Loop (`valid_fn`)
- `torch.no_grad()` inference
- Sigmoid applied: `y_preds.sigmoid()`
- Returns `avg_loss` and stacked `predictions` array

### 9. Cross-Validation Train Loop (`train_loop`)
For each fold:
1. Split data into train/val by fold index
2. Build `DataLoader` objects
3. Instantiate model, optimizer, scheduler
4. Run epoch loop; save two checkpoints per fold:
   - `{model_name}_fold{n}_best_score.pth` (best ROC-AUC)
   - `{model_name}_fold{n}_best_loss.pth` (best validation loss)
5. Load best-loss model at end of fold; store OOF predictions

### 10. Main Function
Iterates over all 4 folds, accumulates OOF predictions, logs per-fold and overall CV ROC-AUC, saves `oof_df.csv`.

---

## Outputs

| File | Description |
|---|---|
| `nfnet_l0_fold{n}_best_score.pth` | Best-AUC model weights per fold |
| `nfnet_l0_fold{n}_best_loss.pth` | Best-loss model weights per fold |
| `oof_df.csv` | Out-of-fold predictions for CV scoring |
| `train.log` | Training log (loss and score per epoch) |

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| Vstack 6 cadence images into 1 | Encodes temporal structure in spatial layout for CNN |
| `in_chans=1` | Spectrograms are single-channel; avoids incorrect RGB assumptions |
| BCEWithLogitsLoss | Numerically stable binary loss; sigmoid applied separately at inference |
| Save both best-score and best-loss checkpoints | Allows post-hoc selection; best-loss used for OOF preds |
| StratifiedKFold | Preserves class balance across folds despite heavy imbalance |

---

## Suggested Improvements

- Add strong augmentations: horizontal flip, ShiftScaleRotate, Cutout, SpecAugment.
- Train with AMP (`CFG.apex=True`) for faster training.
- Ensemble predictions from all 4 folds (average probabilities).
- Try larger models: `nfnet_l1`, `efficientnet_b4`, `convnext_base`.
- Tune image size (e.g., 256×256 or 512×512).
- Add test-time augmentation (TTA) with horizontal flips.
- Use `GroupKFold` on cadence group IDs if leakage between signal sequences is a concern.
