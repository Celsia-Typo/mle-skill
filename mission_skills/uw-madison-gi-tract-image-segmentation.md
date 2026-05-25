1. Problem Understanding

Task type

* Multi-class medical image segmentation on abdominal CT slices
* Segment 3 organs/classes:
    * stomach
    * small bowel
    * large bowel
* 2.5D segmentation pipeline:
    * input consists of neighboring slices stacked as channels
    * typical configurations:
        * slice=3 → [t-1, t, t+1]
        * slice=5 → [t-2 ... t+2]

Evaluation metric

Primary optimization targets:

* Dice coefficient
* Reduction of false positives on empty slices

Competition metric behavior implies:

* Boundary quality matters
* False-positive masks are heavily penalized
* Empty slice handling is critical

Key challenges

* Extremely large number of empty masks
* Small foreground regions
* CT slices contain large irrelevant background regions
* Memory limitations for high-resolution segmentation
* Slice continuity across depth dimension
* Need to suppress noisy predictions

⸻

2. Data Pipeline (Code-Oriented)

load_data()

Responsibilities:

* Read train.csv
* Read image metadata
* Decode RLE masks
* Build dataframe:
    * case id
    * day id
    * slice id
    * image path
    * height/width
    * organ labels
    * empty/non-empty flags

Implementation details:

* Group samples by case
* Sort slices by z-order
* Build neighboring slice index mapping

Output:

df.columns = [
    "id",
    "case",
    "day",
    "slice",
    "image_path",
    "mask_rle",
    "class_name",
    "is_empty"
]

⸻

preprocess()

Step 1 — Detection-based cropping

Use a lightweight detector:

* EfficientDet-D0
* image size = 256

Purpose:

* Detect abdomen ROI
* Remove large black background
* Reduce segmentation memory usage

Pipeline:

bbox = detector.predict(image)
cropped = image[y1:y2, x1:x2]

Training target:

* Bounding boxes around all organs

Implementation:

* Save crop coordinates
* Reuse coordinates during inference

⸻

Step 2 — Build 2.5D samples

For each slice:

channels = [
    slice_{t-k},
    ...,
    slice_t,
    ...,
    slice_{t+k}
]

Boundary handling:

* replicate edge slices

Output shape:

(C, H, W)

where:

* C = 3 or 5

⸻

Step 3 — Resize and normalize

Typical sizes:

* 320
* 352
* 384
* 416

Normalization:

image = image.astype(np.float32) / 255.0

⸻

feature_engineering()

Create:

* positive_flag
* organ pixel counts
* neighboring slice positivity
* case/day grouping ids

Optional:

df["has_mask_neighbor"] = ...

Used for:

* classifier training
* post-processing

⸻

split_folds()

Use:

GroupKFold(n_splits=5)

Group by:

group = case_id

Purpose:

* prevent leakage between slices from same patient

Store:

df["fold"]

⸻

3. Model Design

build_detector()

Model:

* EfficientDet-D0

Configuration:

image_size = 256
epochs = 5

Output:

* bounding box coordinates

Purpose:

* crop abdominal region

⸻

Classification + Segmentation Hybrid

build_model(config)

Architecture:

* UNet decoder
* encoder from timm or segmentation_models_pytorch

Supported backbones:

* efficientnet-b7
* efficientnet-v2-l
* efficientnet-v2-m
* efficientnet-b7ns

Model outputs:

segmentation_logits
classification_logits

⸻

Classification branch

Purpose:

* predict whether slice contains organ mask

Implementation:

cls_head = GlobalPooling + Linear

Loss:

BCEWithLogitsLoss

⸻

Segmentation branch

Decoder:

* UNet

Output channels:

num_classes = 3

Loss combinations:

0.5 * BCE
+ 0.5 * Dice
+ 1.0 * Lovasz

⸻

Pretrained initialization

Use:

* ImageNet pretrained encoders
* timm pretrained weights

Optional:

* multiple pretrained scales/checkpoints

⸻

4. Training Strategy

train_one_fold()

Training stages:

⸻

Stage A — Classification-focused training

Dataset:

* all slices (positive + negative)

Goal:

* maximize empty slice detection

Loss:

cls_loss = BCE
seg_loss = BCE
total_loss = cls_loss + seg_loss

Metrics:

* TP / (TP + FP + FN)
* Dice

⸻

Stage B — Segmentation-focused training

Dataset:

* only positive slices

Goal:

* maximize mask quality

Loss:

ComboLoss = (
    0.5 * BCE +
    0.5 * Dice +
    1.0 * Lovasz
)

⸻

Optimizer

Typical:

AdamW

Learning rate:

3e-4 or 5e-4

Epochs:

35

⸻

Scheduler

Recommended:

CosineAnnealingLR

SWA:

torch.optim.swa_utils

Cycle training:

* 7-cycle schedule

⸻

EMA

Maintain exponential moving average:

ema_model.update(model)

Use EMA weights for validation.

⸻

Augmentations

Useful:

* Mixup
* CutMix

Albumentations:

HorizontalFlip
ShiftScaleRotate
ElasticTransform
GridDistortion

Avoid:

* excessive brightness augmentation

⸻

AMP Training

Use:

torch.cuda.amp.autocast()
GradScaler()

⸻

Gradient clipping

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

⸻

5. Validation Strategy

Cross-validation

Use:

5-fold GroupKFold

Validation performed:

* per fold
* per epoch

⸻

validate()

Compute:

* Dice score
* classification precision/recall
* combined TP/(TP+FP+FN)

Use threshold search:

best_thr = optimize_threshold()

⸻

OOF generation

Store:

oof_masks
oof_cls_probs

Used for:

* ensemble tuning
* threshold calibration

⸻

6. Inference Pipeline

predict_detector()

Steps:

1. Detect ROI
2. Crop image
3. Save bbox

⸻

predict_classifier()

Generate:

cls_prob

If:

cls_prob < threshold

then:

skip segmentation
return empty mask

This significantly reduces false positives.

⸻

predict_segmentation()

Run:

* multiple segmentation models
* different image sizes
* different slice depths

Examples:

* 320/5
* 384/5
* 416/5

⸻

Ensemble

Average:

mean(logits)

Possible weighted ensemble:

0.4 * model_a +
0.3 * model_b +
...

⸻

TTA

Recommended:

* horizontal flip

Implementation:

pred = (pred + flip_pred) / 2

⸻

post_process()

Remove tiny noisy regions

mask = remove_small_objects(mask, min_size=25)

⸻

Temporal consistency filtering

Rule:

* require consecutive positive predictions

Implementation:

start only after 3 consecutive positives
end after 3 consecutive negatives

Purpose:

* stabilize slice continuity

⸻

Resize back to original shape

Use saved crop bbox:

restore_to_original_canvas()

⸻

7. Key Tricks (ACTIONABLE)

Most impactful

1. Detection-based cropping

Huge memory and speed improvement.

Code logic:

crop_before_segmentation = True

⸻

2. Separate classification and segmentation logic

Classifier suppresses empty-slice false positives.

Implementation:

if cls_prob < thr:
    return empty_mask

⸻

3. Train segmentation models only on positive samples

Improves foreground quality significantly.

⸻

4. 2.5D neighboring slices

Use:

slice=5

Better spatial consistency than single-slice training.

⸻

5. EMA + SWA

Improves fold stability and leaderboard consistency.

⸻

Secondary improvements

Mixup and CutMix

Especially effective for segmentation stage.

⸻

Multi-scale ensemble

Use:

* 320
* 384
* 416

⸻

Different backbones

EfficientNet variants ensemble well together.

⸻

Threshold optimization

Tune separately for:

* cls threshold
* mask threshold

⸻

Minor tricks

Remove tiny components

min_size = 25

⸻

Consecutive slice filtering

Small but measurable gain.

⸻

Multiple pretrained checkpoints

Different initialization scales improve diversity.

⸻

8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

import os
import cv2
import gc
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
import albumentations as A
# =========================================================
# CONFIG
# =========================================================
class CFG:
    seed = 42
    num_folds = 5
    image_size = 384
    batch_size = 8
    epochs = 35
    lr = 3e-4
    num_classes = 3
    use_amp = True
    use_ema = True
    use_swa = True
    slice_depth = 5
# =========================================================
# UTILS
# =========================================================
def seed_everything(seed):
    pass
def rle_decode(rle, shape):
    pass
def dice_score(pred, target):
    pass
# =========================================================
# DATA LOADING
# =========================================================
def load_data():
    """
    Load CSV files, metadata, image paths,
    decode masks, build dataframe.
    """
    pass
def preprocess(df):
    """
    ROI crop generation using detector,
    normalization, resize preparation.
    """
    pass
def feature_engineering(df):
    """
    Create empty-mask flags,
    neighboring slice features,
    organ statistics.
    """
    pass
def create_folds(df):
    """
    GroupKFold split by case id.
    """
    pass
# =========================================================
# DETECTOR
# =========================================================
def build_detector():
    """
    EfficientDet-D0 for ROI detection.
    """
    pass
def train_detector():
    pass
def predict_detector():
    pass
# =========================================================
# DATASET
# =========================================================
class GITractDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        pass
    def load_2p5d_image(self, idx):
        """
        Load neighboring slices.
        """
        pass
    def __getitem__(self, idx):
        pass
    def __len__(self):
        return len(self.df)
# =========================================================
# MODEL
# =========================================================
class TimmUnet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
    def forward(self, x):
        """
        Return:
            segmentation_logits,
            classification_logits
        """
        pass
def build_model(cfg):
    """
    Build segmentation/classification model.
    """
    pass
# =========================================================
# LOSSES
# =========================================================
class DiceLoss(nn.Module):
    pass
class LovaszLoss(nn.Module):
    pass
def combo_loss(seg_pred, seg_target):
    pass
# =========================================================
# TRAINING
# =========================================================
def train_one_epoch(model, loader, optimizer, scheduler):
    pass
def valid_one_epoch(model, loader):
    pass
def train_one_fold(fold, df):
    """
    Stage A:
        classification + segmentation
    Stage B:
        segmentation-only positive samples
    """
    pass
# =========================================================
# EMA / SWA
# =========================================================
class ModelEMA:
    pass
def apply_swa(model):
    pass
# =========================================================
# VALIDATION
# =========================================================
def validate(model, loader):
    """
    Compute Dice and cls metrics.
    """
    pass
def optimize_thresholds(oof_preds):
    pass
# =========================================================
# INFERENCE
# =========================================================
def predict_classifier(models, loader):
    pass
def predict_segmentation(models, loader):
    pass
def tta_inference(model, images):
    pass
def ensemble_predictions(preds):
    pass
# =========================================================
# POST PROCESS
# =========================================================
def remove_small_regions(mask, min_size=25):
    pass
def temporal_filtering(preds):
    """
    Consecutive slice smoothing.
    """
    pass
def post_process(mask):
    pass
# =========================================================
# SUBMISSION
# =========================================================
def masks_to_rle(masks):
    pass
def create_submission(preds):
    pass
# =========================================================
# MAIN
# =========================================================
def main():
    seed_everything(CFG.seed)
    df = load_data()
    df = preprocess(df)
    df = feature_engineering(df)
    df = create_folds(df)
    detector = train_detector()
    models = []
    for fold in range(CFG.num_folds):
        model = train_one_fold(fold, df)
        models.append(model)
    test_preds = predict_segmentation(models)
    test_preds = post_process(test_preds)
    submission = create_submission(test_preds)
    submission.to_csv("submission.csv", index=False)
if __name__ == "__main__":
    main()

⸻

9. Strategy Priority (IMPORTANT)

1. Most impactful techniques

1. Detection-based ROI cropping
2. Classification gate before segmentation
3. Positive-only segmentation training
4. 2.5D slice stacking
5. EMA + SWA
6. Multi-model ensemble

⸻

2. Secondary improvements

1. Multi-scale training
2. Mixup/CutMix
3. EfficientNet backbone diversity
4. TTA
5. Threshold tuning

⸻

3. Minor tricks

1. Small object removal
2. Consecutive slice filtering
3. Different pretrained initializations
4. Brightness augmentation avoidance