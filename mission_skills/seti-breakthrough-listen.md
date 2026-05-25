Here is the structured, actionable solution blueprint designed for generating a single-file PyTorch script based on the 1st place SETI Breakthrough Listen writeup. 

## 1. Problem Understanding
* **Task Type:** Binary Image Classification (detecting anomalous E.T. signals in spectrogram arrays).
* **Evaluation Metric:** ROC AUC (standard for this Kaggle competition).
* **Key Challenges:** * Severe domain shift between train and test datasets (different background noise).
    * Test set contains completely novel signal types (e.g., "s-shape" curves) absent in the training data.
    * Extremely low Signal-to-Noise Ratio (SNR).
    * Model overfitting to background noise rather than focusing on the target signals.

## 2. Data Pipeline (Code-Oriented)
* `load_data()`: 
    * Parse CSVs for the "old train", "old test", and "new train" datasets.
    * Concatenate them into a single massive dataframe for training.
* `clean_background_offline()` (Pre-computation step): 
    * Implement an offline algorithm to find matching backgrounds across the dataset using column-wise normalization and difference matching (MSE of the first column). 
    * Create a "cleaned" version of the dataset by replacing original data with the difference of the normed regions between matched samples to induce "shadowing" of the actual signals. Save these to disk.
* `Dataset_Class` (`__getitem__`):
    * Load the `.npy` arrays.
    * **Filter Channels:** Extract *only* the ON-channels (typically channels 0, 2, and 4 in the 6-channel array) and concatenate them into a spatial image.
    * Cast the array to `float32`.
    * Apply channel-wise standardization (mean/std normalization) *per image* immediately after loading.
* `feature_engineering()`: 
    * Implement an "s-shape" signal generator based on `setigen`. 
    * During dataset loading, dynamically inject this artificial s-shape signal into the array with probability `p=0.01`, randomizing shape and SNR.
* `split_folds()`: 
    * Use `StratifiedKFold(n_splits=5)` on the target column, though the final production run will train on all data.

## 3. Model Design
* `build_model(model_name='eca_nfnet_l2')`:
    * **Backbone:** Load `eca_nfnet_l2` via the `timm` library. A norm-free network is strictly required to bypass BatchNorm domain shift issues between train and test sets.
    * **Stride Modification:** Locate the first convolutional layer (stem) and modify its stride from `(2, 2)` to `(1, 2)`. This artificially increases the vertical resolution propagating through the network.
    * **Pooling:** Replace the default pooling layer with a trainable Generalized Mean (GeM) pooling layer.
    * **Head:** Attach a `nn.Linear` layer outputting a single logit.

## 4. Training Strategy
* `train_one_fold()`: 
    * **Epochs & Hardware:** Train for 20 epochs. Use Distributed Data Parallel (DDP) if multiple GPUs are available, else standard PyTorch loop.
    * **Loss Function:** `BCEWithLogitsLoss`.
    * **Optimizer:** `AdamW` or `SGD` with a Cosine Annealing learning rate scheduler.
    * **Input Resolution:** Feed the full, un-resized concatenated array directly into the model.
* `custom_mixup()`: 
    * Apply Mixup inside the training loop (probability = 1.0, drawn from Beta distribution `alpha=5, beta=5`).
    * **Constraint:** Only mix cleaned images with cleaned images, and uncleaned with uncleaned.
    * **Label Logic:** Instead of blending targets linearly, use `target = max(y1, y2)`. If either image has a signal, the mixed image is labeled as 1.
    * **Post-processing:** Re-normalize the image tensor back to zero mean and unit variance *after* the Mixup addition to maintain distribution statistics.

## 5. Validation Strategy
* `cross_validation()`: 
    * Iterate through the 5 folds.
    * Keep track of the CV to LB gap. 
    * Generate Out-Of-Fold (OOF) predictions and calculate ROC AUC.
* `train_full_fit()`: 
    * Since pseudo-labeling and blending are bypassed, implement a flag `TRAIN_ALL_DATA=True`. When active, skip the validation splits and train a single model on 100% of the concatenated data (old + new + test).

## 6. Inference Pipeline
* `predict()`: 
    * For each test ID, check if a "cleaned" array exists. If yes, load it. If no, fallback to the uncleaned array.
* `apply_tta()`: 
    * Implement 4x Test Time Augmentation: `[Original, Horizontal Flip, Vertical Flip, Horizontal+Vertical Flip]`.
    * Pass all 4 through the model, apply sigmoid, and calculate the mean probability.

## 7. Key Tricks (ACTIONABLE)
* **IF** using BatchNorm models **THEN** switch to Norm-Free (`eca_nfnet_l2`) to avoid training dynamics collapsing due to background shifts.
* **IF** performing Mixup on signals **THEN** target must be `max(y_a, y_b)` and the resulting tensor must be channel-normalized *again* post-mix.
* **IF** memory is a constraint **THEN** aggressively drop the OFF channels. The ON channels contain the necessary information.
* **IF** creating the model architecture **THEN** execute `model.conv_stem.stride = (1, 2)`.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import StratifiedKFold
# ... other standard imports ...

def seed_everything(seed=42):
    # Fix all RNG seeds for reproducibility

def get_config():
    # Return a dictionary/class with hyperparameters:
    # epochs=20, backbone='eca_nfnet_l2', mixup_alpha=5.0, etc.

def load_data(config):
    # Load old_train, old_test, new_train CSVs
    # Concatenate into one massive dataframe
    # Return train_df, test_df

def clean_backgrounds(df):
    # Offline step: calculate column-wise distances
    # Subtract overlapping signals
    # Save as separate 'cleaned' .npy files

def generate_s_shape_signal(shape, snr):
    # Use logic inspired by setigen
    # Return a numpy array with an artificial curve

class SETIDataset(Dataset):
    # __init__: handle cleaned vs uncleaned paths
    # __getitem__: 
    #   1. Load array
    #   2. Filter ON channels only & concat
    #   3. Inject s_shape_signal() with p=0.01
    #   4. Cast to float32
    #   5. Apply per-image channel normalization
    #   6. Apply random vertical flip

class GeMPooling(nn.Module):
    # Implement Generalized Mean Pooling with trainable parameter 'p'

def build_model(config):
    # Load timm 'eca_nfnet_l2'
    # Change model.conv_stem.stride to (1, 2)
    # Replace global pool with GeMPooling
    # Replace classifier with nn.Linear(in_features, 1)

def apply_custom_mixup(x, y, alpha=5.0):
    # Draw lambda from Beta(alpha, alpha)
    # mixed_x = lam * x + (1 - lam) * x.flip(dims=[0]) (mix with another in batch)
    # mixed_y = torch.max(y, y.flip(dims=[0]))
    # RE-NORMALIZE mixed_x
    # Return mixed_x, mixed_y

def train_one_fold(fold, train_loader, val_loader, config):
    # Initialize model, AdamW optimizer, Cosine scheduler, BCEWithLogitsLoss
    # For batch in train_loader:
    #   Apply custom_mixup()
    #   Forward pass, backward pass, step
    # Evaluate on val_loader
    # Return trained model

def create_folds(df, n_splits=5):
    # Apply StratifiedKFold
    # Return fold column

def inference_with_tta(model, test_loader):
    # Iterate test loader
    # Apply 4x TTA: raw, hflip, vflip, hvflip
    # Average sigmoids
    # Return predictions

def main():
    config = get_config()
    seed_everything(config.seed)
    
    train_df, test_df = load_data(config)
    
    # Optional offline step (assumed to be pre-run or handled here)
    # clean_backgrounds(train_df) 
    
    train_df = create_folds(train_df)
    
    models = []
    # If full fit: use all data in one fold
    for fold in range(config.n_splits):
        # Extract train/val splits
        # Create DataLoaders
        model = train_one_fold(fold, train_loader, val_loader, config)
        models.append(model)
        
    preds = inference_with_tta(models, test_loader)
    
    # Save submission.csv

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1. **Most Impactful Techniques:**
   * Offline background cleaning (subtracting column-wise overlap distances).
   * Changing the model backbone to `eca_nfnet_l2` (avoiding BatchNorm domain shift constraints).
   * Custom target handling for Mixup (`y = max(y1, y2)`) combined with post-mix tensor re-normalization.

2. **Secondary Improvements:**
   * Injecting artificial `s-shape` signals dynamically into the dataset (`p=0.01`).
   * Altering the first conv-layer stride from `(2,2)` to `(1,2)` to stretch resolution artificially.
   * Training on combined data (Old Train + Old Test + New Train) using a Full Fit.

3. **Minor Tricks:**
   * Dropping OFF channels to reduce memory footprint, allowing for the use of the full image resolution without resizing.
   * Applying trainable GeM Pooling.
   * 4x Test Time Augmentation (TTA).