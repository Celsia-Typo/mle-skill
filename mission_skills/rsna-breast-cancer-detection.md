## 1. Problem Understanding
* **Task Type:** Computer Vision, Binary Classification (Breast Cancer Detection per patient-laterality).
* **Evaluation Metric:** probabilistic F1 score (pF1), heavily dependent on selecting an optimal binarization threshold.
* **Key Challenges:** * Extreme class imbalance (positives are rare).
    * "Needle in a haystack" visual problem (cancerous regions are tiny compared to the whole image).
    * Variable aspect ratios and presence of background noise/artifacts in raw DICOMs.
    * Metric instability (pF1 prioritizes precision over recall, making standard optimization tricky).

## 2. Data Pipeline (Code-Oriented)
* **`load_data()`**: 
    * Read competition CSVs and external dataset metadata (VinDr-Mammo, MiniDDSM, CMMD, CDD-CESM, BMCD).
    * Map external labels to binary targets (e.g., VinDr BIRADS-5 $\rightarrow$ 1, BIRADS-4 $\rightarrow$ 0; MiniDDSM 'Cancer' $\rightarrow$ 1).
* **`preprocess(image_path)`**:
    * Load DICOM/PNG. Apply windowing (VOILUT) if metadata is present.
    * Apply Min-Max normalization (use percentile Min-Max for datasets with watermark noise like MiniDDSM).
    * Execute ROI Crop: Pass image through a pre-trained YOLOX-nano (416x416) to detect breast bounding boxes.
    * Fallback logic: If YOLOX misses, use Otsu thresholding + `cv2.findContours()`. If both fail, return the uncropped image.
    * Resize cropped region to 2048x1024.
* **`feature_engineering(df)`**:
    * Implement a custom batch sampler to control the positive/negative ratio.
    * Enforce a strict constraint: Ensure at least 1 positive sample exists in every batch to stabilize early training. Set the target pos/neg ratio to 1:7.
* **`split_folds(df)`**:
    * Apply 4-fold `StratifiedGroupKFold` grouped by `patient_id` exclusively on competition data.
    * Assign all external datasets to the training set for every fold (do not leak external data into validation).

## 3. Model Design
* **`build_model(config)`**:
    * **Architecture:** Convolutional Neural Network.
    * **Backbone:** `convnext_small.fb_in22k_ft_in1k_384` (loaded via `timm`).
    * **Resolution:** 2048 (height) x 1024 (width).
    * **Pooling:** Implement Global Max Pooling (`nn.AdaptiveMaxPool2d(1)`) instead of Mean or GeM to capture sparse, high-intensity cancer signals.
    * **Head:** Single linear layer outputting 1 logit.
    * **Regularization:** Set `drop_rate=0.5` and `drop_path_rate=0.2`.

## 4. Training Strategy
* **`train_one_fold(fold, train_loader, val_loader, model)`**:
    * **Loss Function:** `BCEWithLogitsLoss` utilizing "Soft Positive Labels". Map negative targets to 0.0 and positive targets to 0.8 or 0.9 (prevents overconfidence on ambiguous patient-level labels).
    * **Optimizer:** SGD with `momentum=0.9`.
    * **Scheduler:** Cosine Annealing LR (`lr=1e-3` or `3e-3`, `min_lr=1e-5` or `5e-5`) coupled with 4 epochs of Linear Warmup.
    * **Tricks:** * Enable PyTorch Automatic Mixed Precision (AMP) via `torch.cuda.amp.autocast`.
        * Utilize Exponential Moving Average (EMA) for model weights to stabilize validation performance.
        * Heavy Augmentations via `albumentations`: RandomSizedCropNoResize, Flips, Random Tone/Brightness, ShiftScaleRotate, GridDistortion, and CoarseDropout (Random Erasing).

## 5. Validation Strategy
* **`validate(model, val_loader)`**:
    * Compute predictions on the validation fold using the EMA model weights.
    * Track multiple metrics simultaneously: PR_AUC, ROC_AUC, and Binarized pF1.
    * **OOF Generation:** Search for the optimal probability threshold (typically between 0.31 and 0.35) by maximizing the binarized pF1 on the Out-Of-Fold predictions.
    * Group image-level predictions by patient-laterality and aggregate using `.mean()`.

## 6. Inference Pipeline
* **`predict(models, test_loader)`**:
    * Execute inference using all 4 fold models. Move array operations to the GPU to circumvent CPU bottlenecks.
    * **Ensemble:** Average the raw sigmoid probabilities across the 4 models.
    * **Post-process:** Group predictions by `patient_id` and `laterality`. Aggregate image scores using `.mean()`.
    * Apply the pre-calculated OOF threshold (e.g., 0.31) to convert continuous probabilities into binary 1/0 predictions.

## 7. Key Tricks (ACTIONABLE)
* **If handling variable breast sizes** $\rightarrow$ Do NOT use standard longest-edge resize with padding. Use YOLOX ROI cropping + `CustomRandomSizedCropNoResize` to maintain scale consistency.
* **If optimizing for "needle in a haystack"** $\rightarrow$ Swap standard global average pooling for Global Max Pooling.
* **If model trains unstably** $\rightarrow$ Hardcode the dataloader to fetch $\ge$ 1 positive sample per batch (batch size = 8).
* **If model is overconfident on weak labels** $\rightarrow$ Implement soft positive labels: `target = 0.9` instead of `1.0`.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc

# --- CONFIGURATION ---
class Config:
    seed = 42
    img_size = (2048, 1024)
    batch_size = 8
    epochs = 24
    backbone = 'convnext_small.fb_in22k_ft_in1k_384'
    soft_pos_label = 0.9
    lr = 1e-3
    min_lr = 1e-5
    warmup_epochs = 4
    n_folds = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- UTILS ---
def seed_everything(seed):
    # Set standard deterministic seeds for reproducibility
    pass

# --- DATA PROCESSING ---
def load_data():
    # Load competition train.csv and external datasets
    # Map external labels, concatenate dataframes
    return train_df, test_df

def preprocess(image_path):
    # Load DICOM/PNG, apply VOILUT / Windowing
    # Run Otsu or YOLOX inference for ROI cropping
    # Resize to Config.img_size and normalize
    return img_array

def create_folds(df):
    # Apply StratifiedGroupKFold on patient_id for competition data
    # Assign external data strictly to training folds
    return df

class MammographyDataset(Dataset):
    def __init__(self, df, transforms=None):
        # Initialize dataframe and albumentations
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        # Call preprocess(), apply transforms, return tensor and soft target
        pass

def get_transforms(is_train):
    # Return A.Compose with custom crops, flips, distortions, and drops for train
    # Return standard normalization for val/test
    pass

# --- MODEL ---
class BreastCancerModel(nn.Module):
    def __init__(self, backbone_name, pretrained=True):
        super().__init__()
        # Load timm model, drop_rate=0.5, drop_path_rate=0.2
        # Replace classifier with AdaptiveMaxPool2d(1) + Linear(1)
        pass
    def forward(self, x):
        pass

# --- TRAINING LOOP ---
def train_one_fold(fold, train_loader, val_loader):
    # Initialize BreastCancerModel
    # Setup BCEWithLogitsLoss, SGD optimizer, CosineAnnealingLR
    # Setup PyTorch AMP scaler and EMA model
    # Loop epochs:
    #   Forward pass, backward pass, optimizer step
    #   Track loss, apply soft positive labels
    #   Call validate() at epoch end
    # Return best_model weights
    pass

def validate(model, val_loader):
    # Run inference without gradients
    # Calculate ROC_AUC, PR_AUC
    # Search for best threshold maximizing pF1
    return metrics_dict, oof_predictions

# --- INFERENCE ---
def inference(models, test_loader):
    # Loop through models and test data
    # Accumulate logits on GPU
    # Apply sigmoid, average across folds
    # Group by patient_id/laterality using mean()
    # Apply optimal threshold -> binary outputs
    return submission_df

def save_submission(df):
    # Format and save to submission.csv
    pass

# --- MAIN EXECUTION ---
def main():
    seed_everything(Config.seed)
    train_df, test_df = load_data()
    train_df = create_folds(train_df)
    
    models = []
    for fold in range(Config.n_folds):
        # Setup specific fold DataLoaders ensuring >=1 pos per batch
        train_loader, val_loader = ... 
        best_model = train_one_fold(fold, train_loader, val_loader)
        models.append(best_model)
    
    if len(test_df) > 0:
        test_loader = ... # Setup test DataLoader
        preds = inference(models, test_loader)
        save_submission(preds)

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)
1.  **Most Impactful Techniques:** External data integration (massive boost to positive sample count) combined with Soft Positive Labels (0.8-0.9) to prevent catastrophic overfitting on ambiguous external data.
2.  **Secondary Improvements:** High-precision ROI extraction using a trained YOLOX-nano bounding box detector to standardize resolution and eliminate background artifacts.
3.  **Minor Tricks:** Global Max Pooling architecture modification and enforcing $\ge$ 1 positive sample per batch via custom samplers to unstick early training dynamics.