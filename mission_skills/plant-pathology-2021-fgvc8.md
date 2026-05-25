## 1. Problem Understanding
* **Task type:** Multi-label Image Classification (identifying one or multiple plant leaf diseases in a single image).
* **Evaluation metric:** Mean F1-Score (macro-averaged across classes), standard for multi-label classification.
* **Key challenges:** * Overlapping/co-occurring labels (e.g., 'scab' and 'frog_eye_leaf_spot' on the same leaf).
    * Noisy annotations in the training data requiring confidence calibration.
    * Class imbalance between common diseases and rare complex combinations.

## 2. Data Pipeline (Code-Oriented)
* **`load_data()`**: Parse the `train.csv`. Initialize global configurations for image paths.
* **`preprocess()`**: Map textual labels into multi-hot binary vectors. For example, 'scab frog_eye_leaf_spot' becomes `[1, 0, 1, 0, ...]`. Treat 'healthy' as an explicit class or an all-zero target vector depending on the network design.
* **`feature_engineering()`**: 
    * Construct the Albumentations pipeline.
    * **Train Augmentations:** `RandomResizedCrop(384, 576)`, `HorizontalFlip`, `VerticalFlip`, `RandomBrightnessContrast`, `ShiftScaleRotate`, `OpticalDistortion`, `GridDistortion`, `IAAPiecewiseAffine` (equivalent to `PiecewiseAffine`), `CoarseDropout` (replacing `Cutout`).
    * **Valid Augmentations:** `Resize(384, 576)`, `Normalize()`.
* **`split_folds()`**: Utilize `MultilabelStratifiedKFold` from the `iterative-stratification` library with `n_splits=5` to ensure balanced multi-label distributions across all folds.

## 3. Model Design
* **`build_model()`**: Instantiate CNN/Transformer architectures using the `timm` library.
* **Model types:** * Stage 1 (Soft Label Generator): `tf_efficientnetv2_s` or `_m`.
    * Stage 2 (Main Ensemble): `resnet50` and `resnext50_32x4d`.
* **Pretrained usage:** Initialize with `pretrained=True` (ImageNet weights). Replace the final classification head with `nn.Linear(in_features, num_classes)`. Do not use a final softmax/sigmoid layer inside the model; output raw logits.

## 4. Training Strategy
* **`train_one_fold()`**: Standard PyTorch training loop but interrupted at the batch level for custom Mixup/Mosaic.
* **Loss function:** `BCEWithLogitsLoss` customized with Label Smoothing.
* **Optimizer / params:** `AdamW` optimizer with weight decay.
* **Scheduler:** `CosineAnnealingLR` combined with a linear Warmup scheduler for the first 1-2 epochs.
* **Tricks (AMP, clipping):** Use PyTorch Native AMP (`GradScaler` and `autocast`) for mixed precision training.

## 5. Validation Strategy
* **Cross-validation logic:** Iterate through the 5 folds. Track validation loss and multi-label F1-score at the end of each epoch. Save the `.pth` file corresponding to the highest validation F1.
* **OOF generation:** Accumulate sigmoid-activated predictions for the validation sets. Concatenate these to calculate the global Out-Of-Fold (OOF) F1-score and search for the optimal binary decision thresholds per class.

## 6. Inference Pipeline
* **`predict()`**: Load test set images via a PyTorch `DataLoader`.
* **TTA / ensemble:** * Implement Test Time Augmentation (TTA). For each image, predict on: Original, Horizontal Flip, and CenterCrop. Average the logits.
    * Ensemble strategy: Average the TTA-adjusted predictions from all 5 folds of EfficientNetV2, ResNet50, and ResNeXt50.
* **post_process()**: Apply the class-specific thresholds found during OOF validation. Convert the binary arrays back to string representations (e.g., if index 0 and 2 are > threshold, output 'scab frog_eye_leaf_spot').

## 7. Key Tricks (ACTIONABLE)
* **Custom Union CutMix/Mosaic:** If applying CutMix, DO NOT interpolate labels (e.g., `lam * y_a + (1-lam) * y_b`). Instead, apply a logical OR/Union via `torch.clamp(y_a + y_b, min=0.0, max=1.0)`. *Logic: A leaf with scab combined with a leaf with frog eye leaf spot results in a leaf with both diseases.*
* **Soft Pseudo-labeling Mitigation:** If training the final models, load OOF predictions from the EfficientNetV2. If `ground_truth == 1` AND `effnet_pred_score < 0.7`, dynamically reassign the target label to `0.3` during the `__getitem__` dataset fetch.
* **Dynamic Class Balancing:** Track the current epoch. If `current_epoch >= total_epochs - 3`, switch the `DataLoader` `sampler` to a `WeightedRandomSampler` that oversamples underrepresented complex disease combinations.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.cuda.amp import autocast, GradScaler

# ==========================================
# Config & Setup
# ==========================================
def seed_everything(seed=42):
    """Fixes random seeds for reproducibility across numpy, random, and torch."""
    pass

# ==========================================
# Data Pipeline
# ==========================================
def load_data(csv_path):
    """Loads CSV, returns raw DataFrame."""
    pass

def preprocess(df):
    """Converts string labels to multi-hot encoded binary columns."""
    pass

def create_folds(df, num_folds=5):
    """Applies MultilabelStratifiedKFold and adds a 'fold' column to the DataFrame."""
    pass

def get_transforms(phase='train'):
    """Returns Albumentations pipeline for train (heavy augs) or valid (resize/norm)."""
    pass

class PlantDataset(Dataset):
    """
    PyTorch Dataset.
    Includes logic for returning soft labels if effnet_pred < 0.7 and target == 1.
    """
    def __init__(self, df, transforms, soft_label_dict=None):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

# ==========================================
# Custom Augmentations (Batch Level)
# ==========================================
def custom_union_cutmix(images, targets, alpha=1.0):
    """
    Performs CutMix on the image tensor. 
    Combines targets using logical OR (torch.clamp(y_a + y_b, 0, 1)).
    """
    pass

def custom_union_mosaic(images, targets):
    """
    Combines 4 images into a grid.
    Combines targets using logical OR.
    """
    pass

# ==========================================
# Model Design
# ==========================================
def build_model(model_name, num_classes, pretrained=True):
    """Loads timm model, replaces classifier head, returns model."""
    pass

# ==========================================
# Training Strategy
# ==========================================
def train_one_fold(fold, train_df, val_df, config):
    """
    Full training loop for a single fold.
    Includes:
    - DataLoader initialization (switching to WeightedRandomSampler for last 3 epochs)
    - CosineAnnealingLR with warmup
    - AMP training step
    - Batch-level custom CutMix probability application
    """
    pass

def validate(model, valid_loader, criterion):
    """Evaluates model on validation set, returning loss and OOF F1 score."""
    pass

# ==========================================
# Inference Pipeline
# ==========================================
def inference(models_list, test_loader):
    """
    Generates predictions using ensemble of models.
    Applies TTA (CenterCrop, HFlip) and averages predictions.
    """
    pass

def post_process_predictions(preds, thresholds, class_names):
    """Applies thresholds to sigmoid probabilities to generate competition string format."""
    pass

def save_submission(results_df, filename="submission.csv"):
    """Writes final dataframe to disk."""
    pass

# ==========================================
# Main Execution
# ==========================================
def main():
    """
    1. seed_everything()
    2. load_data() & preprocess() & create_folds()
    3. Option A: Train EfficientNetV2 to generate soft labels.
    4. Option B: Train ResNet50/ResNeXt50 using soft labels.
    5. Loop through folds -> train_one_fold() -> append to models list.
    6. inference() on test set.
    7. save_submission()
    """
    pass

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1.  **Most impactful techniques:**
    * **Union CutMix & Mosaic:** Standard mixup destroys biological validity for multi-label diseases. Taking the union (logical OR) of the labels accurately reflects co-occurring diseases.
    * **Soft Pseudo-labeling:** Punishing the model less (`0.3` instead of `1.0`) for missing a ground-truth disease that is visually ambiguous (pred < `0.7` by EffNetV2) prevents overfitting to noisy dataset annotations.
2.  **Secondary improvements:**
    * **Ensembling Diverse Architectures:** Combining EfficientNetV2, ResNet50, and ResNeXt50 provides robust generalized features.
    * **Late-Stage Class Balancing:** Switching to a balanced sampler only in the final 3 epochs allows the model to learn general features first, then fine-tune decision boundaries for minority classes.
3.  **Minor tricks:**
    * **Heavy Augmentation Pipeline:** Grid/Optical distortion and PiecewiseAffine prevent overfitting on specific leaf shapes.
    * **TTA (Flip + CenterCrop):** Squeezes out minor fractional improvements on the leaderboard during inference.
    * **Label Smoothing:** standard regularization to prevent overconfidence.