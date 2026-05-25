## 1. Problem Understanding
* **Task type:** Extreme multi-class, fine-grained image classification with a massive number of categories (64,500 classes).
* **Evaluation metric:** Macro F1-Score (standard for Herbarium datasets, treating all classes equally despite severe imbalance).
* **Key challenges:** * **Extreme Long-Tailed Distribution:** A vast majority of classes have very few training samples.
    * **Scale:** The dataset is exceptionally large, severely restricting the number of viable training epochs and hyperparameter search iterations.
    * **Convergence:** Limited compute time prevents full model convergence, demanding aggressive optimization and representation learning techniques from the start.

## 2. Data Pipeline (Code-Oriented)
* **`load_data()`:** Parse the provided JSON metadata (COCO format) to extract image file paths and their corresponding integer category IDs. Construct a centralized Pandas DataFrame with `image_path` and `label` columns.
* **`preprocess()`:** Filter out any corrupted images. Map the 64,500 unique labels to a continuous integer range `[0, 64499]`. Calculate the frequency of each class in the training set to generate a sample count array (critical for the Balanced SoftMax loss in Stage 2).
* **`feature_engineering()`:** Not applicable for standard raw image pipelines, but instantiate a custom PyTorch `Dataset` class that yields `(image_tensor, label)`.
* **`split_folds()`:** Return the entire dataset as a single fold. Given the massive dataset size and empirically proven lack of overfitting (up to 40 epochs), allocate 100% of the data to training to maximize model exposure to rare classes.

## 3. Model Design
* **`build_model(backbone_name)`:**
    * **Base Architectures (via `timm`):** Initialize `tresnet_m`, `tresnet_l`, `genet_large`, or `eca_nfnet_l0` with pretrained weights.
    * **Custom Classification Head:** Strip the original classifier and attach a multi-stage projection head designed for joint metric and classification learning:
        * `Backbone Feature Extractor` -> `GlobalAveragePooling1D`
        * `Linear Layer 1 (in_features, 2048)` -> `BatchNorm1d` -> `LeakyReLU`
        * `Linear Layer 2 (2048, 512)` -> `BatchNorm1d` -> `LeakyReLU` *(Output A: 512-dim embedding for Metric Learning)*
        * `Linear Layer 3 (512, 64500)` *(Output B: 64500-dim logits for Classification)*
    * **Return Format:** The `forward` pass must return both `Output A` and `Output B` during training.

## 4. Training Strategy
* **`train_one_fold()`:** Implement a two-stage training loop totaling 35 epochs.
    * **Stage 1 (Epochs 1-30):** * Compute **SoftTriple Loss** on the 512-dim embedding (`Output A`).
        * Compute standard **CrossEntropy Loss** on the 64500-dim logits (`Output B`).
        * Total Loss = `SoftTriple(Output A) + CrossEntropy(Output B)`.
    * **Stage 2 (Epochs 31-35):**
        * Compute **SoftTriple Loss** on `Output A`.
        * Compute **Balanced SoftMax Loss** on `Output B` (using the class frequencies calculated during preprocessing to penalize head classes and boost tail classes).
        * Total Loss = `SoftTriple(Output A) + BalancedSoftMax(Output B)`.
* **Optimizer:** Use `timm.optim.AdamP` with a learning rate of `0.001` and `weight_decay=0`.
* **Scheduler:** Apply `CosineAnnealingLR` spanning the full 35 epochs.
* **Tricks:**
    * Wrap training in PyTorch Automatic Mixed Precision (`torch.cuda.amp.autocast` and `GradScaler`) to handle the heavy memory overhead of 64.5k classes.
    * Wrap the model in `ModelEmaV2` (Exponential Moving Average) to stabilize the weights for final inference.
    * **Augmentations (Albumentations):** `RandomResize` -> `RandomCrop(448x448)` -> `HorizontalFlip(p=0.5)` -> `RandomAugMix(severity=3, width=3)` -> `Cutout(p=0.5)` -> `ToTensorV2`. Ensure AugMix is applied without JSD loss constraints.

## 5. Validation Strategy
* **Cross-Validation Logic:** Skip standard validation. The solution blueprint will define a `validate()` function structurally, but the execution pipeline will bypass it, training on 100% of the available data to maximize metric performance on the private leaderboard.
* **OOF Generation:** None required. Predictions are generated directly on the unseen test set.

## 6. Inference Pipeline
* **`predict()`:** Implement a 5-iteration Test-Time Augmentation (TTA) loop.
    * TTA Pipeline: `RandomResize` -> `RandomCrop(448x448)` -> `HorizontalFlip(p=0.5)` -> `Normalize`.
    * Average the softmax probabilities across the 5 TTA passes for each model.
* **Ensemble Strategy:** Perform soft-voting (averaging the TTA-adjusted probabilities) across the diverse backbones (TResNet variants, GENet, NFNet) to yield final `top_1_conf`, `top_1_class`, `top_2_conf`, and `top_2_class` arrays.
* **`post_process()`:** Execute a custom heuristic to curb overconfident predictions on dominant classes. Track the global frequency of `top_1_class` predictions. Iterate through test samples and swap the top 1 and top 2 predictions if:
    * *Condition A:* `top_2_conf >= (top_1_conf * 0.7)`, AND `top_2_class` hasn't been predicted as a top-1 class yet, AND `top_1_class` has been predicted more than twice globally.
    * *Condition B:* `top_2_conf >= (top_1_conf * 0.6)`, AND `top_2_class` hasn't been predicted as a top-1 class yet, AND `top_1_class` has been predicted more than 15 times globally.

## 7. Key Tricks (ACTIONABLE)
* **Metric + Classification Joint Learning:** If building the model, route the penultimate layer to `SoftTripleLoss` and the final layer to `CrossEntropy/BalancedSoftMax`.
* **Long-Tail Handling:** If epoch > 30, switch the classification loss criterion from `nn.CrossEntropyLoss` to a custom `BalancedSoftMaxLoss` initialized with the dataset's class distribution array.
* **Image Dimensions:** Force `img_size = 448` across all architectures to balance resolution with memory constraints.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
from timm.utils import ModelEmaV2

# ==========================================
# 1. Configuration & Utilities
# ==========================================
class CFG:
    # Hyperparameters, paths, and toggles
    pass

def seed_everything(seed):
    # Fix random seeds for reproducibility
    pass

# ==========================================
# 2. Data Pipeline
# ==========================================
def load_data(json_path):
    # Parse COCO JSON to DataFrame
    pass

def preprocess(df):
    # Calculate class frequencies for Balanced SoftMax
    pass

class HerbariumDataset(Dataset):
    # Custom PyTorch Dataset yielding images and targets
    pass

def get_train_transforms():
    # Albumentations: Resize, Crop, HFlip, AugMix, Cutout
    pass

def get_test_transforms():
    # Albumentations: Resize, Crop, HFlip, Normalize
    pass

# ==========================================
# 3. Model Architecture & Losses
# ==========================================
class CustomLossHead(nn.Module):
    # SoftTriple Loss + CrossEntropy/BalancedSoftMax implementations
    pass

class HerbariumModel(nn.Module):
    # timm backbone + custom multi-layer FC head (2048 -> 512 -> 64500)
    pass

# ==========================================
# 4. Training Core
# ==========================================
def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, scaler, ema, epoch):
    # Stage 1/Stage 2 logic branching, AMP forward/backward passes
    pass

def validate(model, dataloader):
    # Structural placeholder (skipped in final run)
    pass

# ==========================================
# 5. Inference & Post-Processing
# ==========================================
def predict_with_tta(model, test_loader, num_tta=5):
    # Run test loader 5 times with random transforms, average probs
    pass

def apply_post_process(predictions):
    # Heuristic swapping of top-1 and top-2 based on confidence and frequency
    pass

# ==========================================
# 6. Main Execution
# ==========================================
def main():
    # 1. Setup & Data Loading
    # 2. Init Model, EMA, Optimizer (AdamP), Scheduler (Cosine)
    # 3. For epoch in range(35):
    #      if epoch == 30: switch_to_balanced_softmax()
    #      train_one_epoch()
    # 4. For each model in ensemble:
    #      predict_with_tta()
    # 5. Ensemble soft voting
    # 6. apply_post_process()
    # 7. Generate submission.csv
    pass

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1.  **Most impactful techniques:** The custom network head combining Metric Learning (SoftTriple) on the 512-dim layer with Classification Learning (Balanced SoftMax) on the 64.5k-dim layer. This directly tackles the massive class count and long-tail distribution.
2.  **Secondary improvements:** Heavy Ensembling (5 diverse large models) combined with 5-fold Test-Time Augmentation (TTA). This squeezes maximum performance out of models that haven't fully converged.
3.  **Minor tricks:** The post-processing confidence switching logic to aggressively penalize over-predicted dominant classes, and training on 100% of the data without a validation split.