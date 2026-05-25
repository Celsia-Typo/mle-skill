Here is the structured, code-oriented blueprint designed to guide an LLM in generating a complete, single-file Kaggle solution. 

## 1. Problem Understanding
* **Task Type:** 3D Medical Image Segmentation (Stage 1) followed by Multi-label Sequential Binary Classification (Stage 2).
* **Evaluation Metric:** Weighted Multi-Label Logarithmic Loss (standard competition metric for RSNA Cervical Spine).
* **Key Challenges:** * Handling massive 3D CT scan data within strict GPU memory constraints.
    * Isolating individual vertebrae accurately to identify highly localized fractures.
    * Mapping vertebrae-level features to patient-level overall outcomes without destroying sequential context.

## 2. Data Pipeline (Code-Oriented)
The pipeline operates in two distinct phases: segmentation data prep and classification data prep.

* **`load_data()`:** Parse CSV metadata and load DICOM image directories. Map `StudyInstanceUID` to actual file paths. Implement a Dicom-to-Numpy parser that sorts slices by Z-axis position.
* **`preprocess_stage1()`:** Resize whole 3D volumes to `128x128x128` using trilinear interpolation. Apply standard Hounsfield Unit (HU) windowing for bone structures.
* **`feature_engineering_stage2()`:** * *Bounding Box Extraction:* Using Stage 1 mask outputs, calculate 3D bounding boxes for C1 through C7 vertebrae.
    * *2.5D Slicing:* For each cropped vertebra, select 15 equidistant slices along the Z-axis.
    * *Channel Stacking:* For each of the 15 slices, stack it with 2 adjacent slices above and 2 below to create a 5-channel 2.5D image. 
    * *Mask Injection:* Append the corresponding binary mask slice as the 6th channel. Output shape per vertebra sample: `(15, 6, H, W)`.
* **`split_folds()`:** Implement `GroupKFold` (k=5) grouped by `StudyInstanceUID` to ensure no patient leaks across train/validation splits.

## 3. Model Design
The solution relies on the `timm` library for 2D backbones and PyTorch for 3D/Sequential heads.

* **`build_seg_model(config)`:** * *Architecture:* 3D UNet with 3D-adapted encoders (ResNet18d or EfficientNetV2-S).
    * *Output:* 7 channels (one for each C1-C7 vertebra).
* **`build_cls_model_type1(config)`:** (Vertebra-Level)
    * *Input Shape:* `(Batch, 15, 6, H, W)`.
    * *Architecture:* TimeDistributed 2D CNN (EfficientNetV2-S or ConvNeXt-Tiny) to extract features per slice $\rightarrow$ Bi-directional LSTM $\rightarrow$ Linear layer outputting fracture probability for a single vertebra.
* **`build_cls_model_type2(config)`:** (Patient-Level)
    * *Input Shape:* `(Batch, 105, 6, H, W)` [7 vertebrae $\times$ 15 slices].
    * *Architecture:* Identical structure to Type 1, but processes all 105 slices sequentially to output the `patient_overall` fracture probability. Uses smaller backbones (ConvNeXt-Nano/Pico/Tiny or NFNet-L0) due to memory limits.

## 4. Training Strategy
* **`train_one_fold(fold, mode)`:** Handles logic based on whether we are training the Seg, Type1, or Type2 model.
* **Loss Function:** `BCEWithLogitsLoss`. For Stage 2, apply competition-specific positive/negative sample weights.
* **Optimizer:** `AdamW` with Cosine Annealing Learning Rate Scheduler.
* **Actionable Tricks:**
    * *Memory Management:* Force `batch_size=1` for Type 2 models.
    * *Speed/Stability:* Utilize Automatic Mixed Precision (`torch.cuda.amp.autocast`) and Gradient Accumulation steps (e.g., `accumulate_steps=8`) to simulate larger batch sizes. Apply gradient clipping (`max_norm=1.0`).

## 5. Validation Strategy
* **Cross-Validation:** Iterate through 5 folds, saving `best_loss.pth`.
* **OOF Generation:** Collect Out-Of-Fold predictions across all batches and concatenate them. Calculate the competition-specific weighted log loss locally to correlate with the public leaderboard.

## 6. Inference Pipeline
* **`predict_segmentation()`:** Run the 3D CT scan through the Stage 1 ensemble, threshold outputs, and generate hard masks.
* **`crop_and_prep_inference()`:** Execute the 2.5D 6-channel extraction dynamically in memory.
* **`predict_classification()`:**
    * Pass the 15-slice blocks through Type 1 models to get vertebrae probabilities.
    * Pass the 105-slice blocks through Type 2 models to get patient overall probabilities.
    * Average predictions across fold models.
* **`post_process()`:** Format predictions into the required `row_id` and `fractured` target column format.

## 7. Key Tricks (ACTIONABLE)
* **If 3D CNNs fail on sparse signals $\rightarrow$ Do 2.5D Sequential Modeling.** Extract slices, pack depth into channels, and use an LSTM for spatial context.
* **If multiple vertebrae overlap in a crop $\rightarrow$ Do Mask Injection.** Append the predicted mask as an extra input channel so the CNN attends *only* to the target bone.
* **If patient-level context is lost by evaluating single bones $\rightarrow$ Do Macro-Sequence Modeling.** Feed all 105 slices of a patient into a secondary model (Type 2) specifically optimized for the `patient_overall` target.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)
This skeleton provides the exact functional layout required for an LLM to populate a monolithic Python execution script.

```python
import os, sys, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
import timm

# --- CONFIGURATION ---
class Config:
    seed = 42
    stage = "inference" # Options: 'train_seg', 'train_cls1', 'train_cls2', 'inference'
    # Add hyperparameters here...

# --- UTILS ---
def seed_everything(seed):
    # Sets seeds for torch, numpy, os, random

# --- DATA PROCESSING ---
def load_data(meta_csv, dicom_dir):
    # Reads dataframe, maps UIDs to image paths

def preprocess_stage1_3d(image_paths):
    # Loads DICOMs, resizes to 128x128x128, applies HU windowing

def feature_engineering_stage2_25d(volume_3d, mask_3d):
    # Extracts bounding boxes, generates 15 slices * 6 channels (5 image + 1 mask)

def create_folds(df):
    # Returns df with a 'fold' column using GroupKFold

# --- DATASETS ---
class RSNA_Seg_Dataset(Dataset):
    # Returns 128x128x128 volumes and 7-ch 3D masks

class RSNA_Cls_Dataset(Dataset):
    # Returns (15, 6, H, W) for Type1 or (105, 6, H, W) for Type 2

# --- MODELS ---
class SegModel3D(nn.Module):
    # 3D UNet with timm encoders

class ClsModelType1(nn.Module):
    # 2D timm CNN feature extractor -> TimeDistributed -> LSTM (Vertebra level)

class ClsModelType2(nn.Module):
    # 2D timm CNN feature extractor -> TimeDistributed -> LSTM (Patient level)

# --- TRAINING LOOP ---
def train_one_fold(fold, model_type, train_df, val_df, config):
    # Init dataloaders, model, AdamW, loss, AMP scaler
    # Loop epochs, log val loss, save best model weights
    return best_model_path

# --- VALIDATION ---
def validate(model, val_loader):
    # Returns raw predictions and loss for OOF evaluation

# --- INFERENCE PIPELINE ---
def run_stage1_inference(seg_models, test_df):
    # Generates and caches 3D masks for all test subjects
    return mask_dictionary

def run_stage2_inference(cls1_models, cls2_models, test_df, mask_dictionary):
    # Crops 2.5D data dynamically, runs sequence models, averages fold predictions
    return final_predictions

# --- MAIN EXECUTION ROUTINE ---
def main():
    cfg = Config()
    seed_everything(cfg.seed)
    
    df, dicom_dir = load_data(...)
    df = create_folds(df)
    
    if cfg.stage == "train_seg":
        # loop train_one_fold for SegModel3D
    elif cfg.stage == "train_cls1":
        # loop train_one_fold for ClsModelType1
    elif cfg.stage == "train_cls2":
        # loop train_one_fold for ClsModelType2
    elif cfg.stage == "inference":
        # 1. Load trained models
        # 2. masks = run_stage1_inference(...)
        # 3. preds = run_stage2_inference(...)
        # 4. save submission.csv

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1.  **Most Impactful Technique:** Transitioning from pure 3D CNNs to 2.5D Convolution + LSTM sequences, solving the sparse spatial signal problem.
2.  **Secondary Improvement:** The Dual-Model Architecture (Type 1 for individual vertebrae, Type 2 for entire patient).
3.  **Minor Tricks:** Appending the semantic mask as an explicit 6th input channel to naturally force the CNN attention onto the target vertebrae during classification.