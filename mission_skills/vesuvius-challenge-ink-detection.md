## 1. Problem Understanding
* **Task type:** 3D Volume to 2D Binary Segmentation (Medical/CT scan style).
* **Evaluation metric:** Expected to be F0.5 score (heavily weighted towards precision), evaluated per pixel.
* **Key challenges:** * Varying depth of the target (ink) across different fragments.
    * Massive image sizes requiring heavy cropping and memory management.
    * High risk of overfitting to the small public leaderboard.
    * Calibration of prediction thresholds across different models.

## 2. Data Pipeline (Code-Oriented)
* **`load_data()`**: 
    * Read the 3D volume as arrays (e.g., Zarr or TIFF stacks). 
    * Extract strictly the middle 16 layers of the Z-axis to standardize input depth.
    * Load corresponding 2D label masks.
* **`preprocess()`**: 
    * Slice the massive volumes into `1024x1024` overlapping spatial crops.
    * **Critical Filter:** Compute the sum/variance of the label mask for each crop. Discard completely blank/empty crops during training to massively accelerate epoch speed.
* **`get_transforms()`**: 
    * Implement an Albumentations pipeline. 
    * *Core:* `VerticalFlip(p=0.5)`, `HorizontalFlip(p=0.5)`, `RandomRotate90(p=0.75)`.
    * *Secondary:* `RandomBrightnessContrast(p=0.5)`, `ShiftScaleRotate(p=0.1)`.
    * *Noise/Distortion:* `GaussNoise(p=0.1)`, `Blur(p=0.1)`, `GridDistortion(p=0.1)`, `CoarseDropout(p=0.1)`.
    * *Custom:* Implement a 1-2 channel (depth layer) dropout with `p=0.25` to force depth invariance.
* **`split_folds()`**: 
    * Use spatial chunking or group by fragment. For local validation, rigidly isolate Fragment 1 as the hold-out set, while training on the rest.

## 3. Model Design
* **`build_model()`**: Construct a custom `nn.Module` containing a two-stage architecture to achieve depth-invariance.
* **Stage 1: 3D Feature Extractor**
    * *Input:* `(Batch, 1, 16, 1024, 1024)`
    * *Backbone:* 3D U-Net, 3D CNN (4 layers increasing filters), or 3D UNETR.
    * *Output:* `(Batch, Channels [16, 32, or 64], 16, 1024, 1024)`
* **The Bridge (Dimensionality Reduction)**
    * Apply standard Dropout.
    * Apply `torch.max(dim=2)` (Max Pooling across the Z/depth axis). 
    * *New Shape:* `(Batch, Channels, 1024, 1024)`.
* **Stage 2: 2D Segmentation Decoder**
    * *Input:* The pooled feature map.
    * *Backbone:* HuggingFace `SegformerForSemanticSegmentation` (variants `b3` to `b5`).
    * *Output:* Segformer natively outputs `(Batch, 1, 256, 256)`.
* **Upsampling Head**
    * Apply a simple `nn.ConvTranspose2d` block to upscale the `256x256` output back to the original `1024x1024` resolution.

## 4. Training Strategy
* **`train_one_fold()`**: 
    * **Loss function:** Linear combination of `BCEWithLogitsLoss` + `DiceLoss`.
    * **Optimizer:** `AdamW` with careful minimum learning rate boundaries.
    * **Scheduler:** Linear warmup followed by Cosine Annealing.
    * **Tricks:** * Enable PyTorch Automatic Mixed Precision (AMP) via `GradScaler` to fit the large `1024x1024` + 3D tensors into VRAM.
        * Implement Stochastic Weight Averaging (SWA) via `torch.optim.swa_utils` to smooth the loss landscape and avoid picking a brittle single checkpoint.
    * **Phase 2 Training:** After validating hyperparameters on Fragment 1, retrain the model against *all* available data (Fragments 1, 2, 3) for the final weights.

## 5. Validation Strategy
* **`evaluate_model()`**: 
    * Run inference on the hold-out validation set (Fragment 1).
    * **Metric Sweeping:** Iterate through probability thresholds (e.g., 0.3 to 0.7 in 0.05 steps) and calculate AUC, Precision, Recall, and F0.5. 
    * Store the optimal threshold locally to verify model calibration.

## 6. Inference Pipeline
* **`predict()`**: 
    * Iterate over the test volumes using a sliding window approach with a stride equal to 1/4 of the crop size (e.g., 256 pixels for a 1024 crop).
    * **Test Time Augmentation (TTA):** Predict 4 times per patch using 90-degree rotations.
    * **Aggregation Math:** Due to TTA and striding, each pixel gets ~16 predictions per model. Average these predictions *as raw logits/pre-sigmoid values* first. Then apply the `sigmoid()` activation.
    * **Ensemble Accumulation:** Average the final sigmoid probabilities across the 9 distinct model architectures. Allocate tensors on CPU incrementally to avoid Kaggle RAM limits.
* **`post_process()`**: 
    * Apply a global threshold of `0.5` (calibrated by the large ensemble).
    * Run `cv2.connectedComponentsWithStats`.
    * Filter out and delete any positive pixel masses smaller than `10,000` pixels (noise reduction).

## 7. Key Tricks (ACTIONABLE)
* **IF** memory allows during training **THEN** increase crop size up to `1024x1024`. Larger spatial context helps the model recognize full character structures.
* **IF** predicting over sliding windows with TTA **THEN** average the predictions in logit space before applying the sigmoid function for better mathematical stability.
* **IF** an ensemble model dictates an optimal threshold wildly different from `0.5` **THEN** include it anyway; averaging predictions across diverse models naturally pulls the optimal global threshold to `0.5`.
* **IF** creating the 3D-to-2D bridge **THEN** always use max-pooling along the Z-axis rather than 2D slicing, preventing the model from receiving false negative signals regarding exactly which depth layer the ink resides in.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import gc
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
# ... other standard imports ...

def seed_everything(seed=42):
    """Locks all random seeds for PyTorch, NumPy, and Python for reproducibility."""
    pass

def load_data(data_dir, is_train=True):
    """Loads middle 16 Z-slices of 3D TIFFs and corresponding 2D masks into memory."""
    pass

def get_transforms():
    """Returns Albumentations pipeline: H/V flips, 90-rotations, brightness, and custom channel dropout."""
    pass

class VesuviusDataset(Dataset):
    """
    PyTorch Dataset. 
    - Slices 1024x1024 crops. 
    - Implements the 'blank crop filter' in __init__ to discard empty masks.
    """
    pass

class Stage1_3D(nn.Module):
    """Implements 3D UNETR, 3D UNet, or 3D CNN block to extract volumetric features."""
    pass

class Stage2_2D(nn.Module):
    """Wraps HuggingFace Segformer b3/b5 for 2D semantic segmentation."""
    pass

class InkDetector(nn.Module):
    """
    Combines Stage1 and Stage2. 
    - Passes input through Stage1.
    - Applies torch.max(dim=2) for depth pooling.
    - Passes to Stage2.
    - Applies ConvTranspose2d to upscale 256x256 -> 1024x1024.
    """
    pass

def train_one_fold(model, train_loader, val_loader, config):
    """
    Executes training loop:
    - AdamW + Cosine Warmup.
    - Mixed Precision (AMP).
    - Accumulates SWA weights in later epochs.
    - Sweeps thresholds to calculate best F0.5 score per epoch.
    """
    pass

def inference_sliding_window(models, volume, config):
    """
    Executes inference with 1/4 stride overlap and 4x Rotation TTA.
    - Averages ~16 logits per pixel.
    - Applies Sigmoid.
    - Averages probabilities across all 'models'.
    - Yields memory-efficient CPU accumulation.
    """
    pass

def post_process(prediction_mask, min_size=10000):
    """Applies threshold (0.5) and removes small blobs using cv2.connectedComponents."""
    pass

def main():
    """
    Main execution controller:
    1. Parses config.
    2. Builds Datasets and Loaders.
    3. Triggers train_one_fold for model variants.
    4. Triggers inference_sliding_window for test data.
    5. Runs post_process.
    6. Saves submission.csv.
    """
    pass

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1.  **Most Impactful Techniques:** * 3D-to-2D depth-invariant architecture (Max-pooling the Z-axis).
    * Large input crops (`1024x1024`) combined with dropping empty masks.
    * Massive Ensembling (9 models) resolving the calibration/thresholding instability.
2.  **Secondary Improvements:** * Test Time Augmentation (4x Rotations) combined with 1/4 strided overlap averaging.
    * Stochastic Weight Averaging (SWA) to widen local optima.
3.  **Minor Tricks:** * Blob removal post-processing (`< 10k` pixels).
    * Depth-channel dropout during augmentation.