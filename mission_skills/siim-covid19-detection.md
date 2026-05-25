Here is the structured, actionable blueprint to guide the generation of a single-file Kaggle solution based on the 3rd place architecture.

## 1. Problem Understanding
* **Task Type:** Dual-track Computer Vision (Multi-label classification at the study level; Object detection at the image level).
* **Evaluation Metric:** Mean Average Precision (mAP) for both study-level classes and image-level bounding boxes.
* **Key Challenges:** High imbalance, visually subtle medical anomalies, aligning image-level bounding boxes with study-level diagnostic classifications, and bridging the domain gap from standard ImageNet weights to chest X-rays.

## 2. Data Pipeline (Code-Oriented)
* **`load_data()`**: Parse DICOM metadata and competition CSVs. Extract bounding box coordinates and target strings. Convert DICOM arrays to 8-bit PNGs to standardize the input for standard CV pipelines.
* **`preprocess()`**: 
    * Append a 'none' class to the 4 default study labels, creating a 5-class target vector.
    * Generate pixel-level segmentation masks by drawing filled ellipses inside the ground-truth bounding boxes.
    * Implement an ROI (Region of Interest) cropper: isolate areas where detection confidence > 0.3 or a segmentation mask exists. Pad this cropped area by 100 pixels globally to preserve contextual boundaries (e.g., pleural effusions).
* **`feature_engineering()`**: Handle the unique 1-image-per-study mapping. Aggregate bounding boxes to the image level.
* **`split_folds()`**: Generate 5-fold stratified splits using the 5-class study labels to ensure uniform class distribution across validation sets.

## 3. Model Design
* **`build_model()`**: Instantiate a branching architecture.
    * **Type A (Pure Classifier):** Swin Transformer base. Input resolution 384x384. Head modified for 5-class multi-label output.
    * **Type B (Hybrid Class-Seg):** Built via `segmentation_models_pytorch`. Encoder: EfficientNet-B6 or Swin FPN. Input resolution 512x512. Dual heads: one for 5-class pooling, one for 1-class (opacity) pixel-wise segmentation.
    * **Type C (Detectors):** YOLOv5 (l6/l), EfficientDet (D6 at 640x640, D7X at 512x512), and Swin-RepPoints via MMDetection.
* **Pretrained Usage:** *Critical step.* Initialize weights sequentially: ImageNet → RSNA 2018 Pneumonia → CheXpert → Current Competition.

## 4. Training Strategy
* **`train_one_fold()`**: Standard PyTorch training loop with mixed precision (AMP).
* **Loss Function**: 
    * Study level: `BCEWithLogitsLoss`. For a subset of models, set the loss weight of negative (normal) images to 0 to force learning on anomalies.
    * Hybrid models: `BCE` for classification + `BCE` for segmentation, summed with a 1:1 ratio.
* **Optimizer / Params**: AdamW optimizer. 
* **Tricks**: 
    * **Augmentations:** Albumentations pipeline (RandomContrast, RandomBrightness, GaussNoise, ShiftScaleRotate).
    * **Mixup:** Apply standard Mixup (alpha=0.5) to the input batches and target vectors.
    * **Data Filtering (Detectors):** Train object detectors exclusively on positive (opacity-containing) images.

## 5. Validation Strategy
* **Cross-Validation Logic**: Evaluate the 5-class study mAP and bounding box mAP iteratively per fold. 
* **OOF Generation**: Store Out-Of-Fold predictions for both classification probabilities and bounding boxes to compute a single, unified local CV score before ensembling.

## 6. Inference Pipeline
* **`predict()`**: Run test images through the 5-fold ensemble of all model types.
* **TTA / Ensemble**: 
    * Classifiers: Simple mathematical averaging across the 10 models (5-fold Swin + 5-fold EffNet-Hybrid).
    * Detectors: 3x Multi-scale Test Time Augmentation (512 to 768).
    * Box Fusion: Use Weighted Box Fusion (WBF) with an IoU threshold of 0.6 to merge YOLO, EffDet, and MMDet predictions.
* **`post_process()`**: Execute domain-specific rule adjustments.
    * Penalty equation: `adjusted_det_conf = original_det_conf * (1 - image_none_prediction)**0.4`
    * None prediction calibration: `final_none_pred = (classification_none_pred**0.7) + (1 - max_image_conf)**0.3`

## 7. Key Tricks (ACTIONABLE)
* **If** training the initial backbone, **→ do** utilize progressive pre-training (RSNA 2018 -> CheXpert) before training on the specific target data.
* **If** cropping images for the classifier, **→ do** pad the bounding box by exactly 100 pixels to avoid stripping contextual features.
* **If** training the segmentation branch, **→ do** convert square bounding boxes into ellipses for the mask generation; it matches lung opacity shapes better than rectangles.
* **If** predicting bounding boxes, **→ do** penalize the bounding box confidence using the study-level "none" (healthy) probability.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
# Note: Assume timm, segmentation_models_pytorch, and ensemble_boxes are available

def seed_everything(seed=42):
    # Lock all RNG seeds for reproducibility
    pass

def load_data(csv_path, img_dir):
    # Parse CSV, extract bboxes, map to DICOM/PNG paths
    pass

def preprocess(df):
    # Add 'none' class to targets
    # Convert string bboxes to numerical lists
    pass

def generate_seg_mask(bboxes, image_shape):
    # Create empty mask, draw filled ellipses based on bboxes
    pass

def apply_roi_crop(image, mask, padding=100):
    # Find mask contours, calculate bounding rect, add 100px padding, crop image
    pass

def create_folds(df, n_splits=5):
    # Apply StratifiedKFold on the 5-class target, return fold indices
    pass

def get_transforms(phase):
    # Return Albumentations pipeline (contrast, brightness, noise, affine) for train/valid
    pass

class CovidDataset(Dataset):
    # Handle image loading, cropping, mixup logic, mask generation, and transforms
    pass

class HybridModel(nn.Module):
    # Build SMP model (EffNet-B6 or Swin FPN)
    # Forward pass returns tuple: (class_logits, seg_mask_logits)
    pass

def criterion(class_preds, class_targets, seg_preds, seg_targets):
    # Compute BCEWithLogits for classes
    # Compute BCE for segmentation
    # Return 1:1 weighted sum
    pass

def train_one_fold(fold, train_loader, val_loader, model, optimizer, scaler):
    # Standard PyTorch loop with AMP
    # Implement zero-loss mask for negative images inside batch loop
    # Save best weights based on validation loss
    pass

def validate(model, val_loader):
    # Generate OOF predictions for fold
    pass

def predict_study_level(models, test_loader):
    # Inference for classifiers, return simple average probabilities
    pass

def predict_image_level_boxes(yolo_models, effdet_models, test_loader):
    # Inference for detectors with 3x multi-scale TTA
    # Apply WBF (IoU=0.6)
    pass

def post_process_predictions(study_preds, box_preds):
    # adjusted_conf = conf * (1 - study_none_pred)**0.4
    # final_none = study_none**0.7 + (1 - max_box_conf)**0.3
    pass

def main():
    seed_everything(42)
    
    # 1. Pipeline execution
    df = load_data('train.csv', 'images/')
    df = preprocess(df)
    df = create_folds(df)
    
    # 2. Classifier Training Loop
    classifier_models = []
    for fold in range(5):
        # Init datasets/loaders
        # Init HybridModel with RSNA->CheXpert pre-trained weights
        # Train & Append best model to list
        pass
        
    # 3. Detector Training Loop (Abstracted, assumes standard pos-only loop)
    # detector_models = train_detectors(df) 
    
    # 4. Inference
    # test_df = load_data('sample_submission.csv', 'test_images/')
    # study_preds = predict_study_level(classifier_models, test_loader)
    # box_preds = predict_image_level_boxes(detector_models, test_loader)
    
    # 5. Post-process & Save
    # final_preds = post_process_predictions(study_preds, box_preds)
    # save_submission(final_preds)

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1.  **Most Impactful Techniques:** * Domain-specific sequential pre-training (RSNA 2018 → CheXpert → Competition Data).
    * Hybrid classification-segmentation architecture forcing the model to learn spatial features.
    * Cross-task post-processing (adjusting box confidence via study-level predictions).
2.  **Secondary Improvements:** * ROI cropping based on bounding boxes/masks padded by 100 pixels.
    * Generating segmentation labels as ellipses rather than raw rectangular bounding boxes.
    * Weighted Box Fusion (WBF) ensemble of diverse detector architectures.
3.  **Minor Tricks:** * Mixup augmentation (0.5).
    * Setting negative image loss to zero for a subset of the models to boost diversity.
    * Albumentations scaling/shifting.