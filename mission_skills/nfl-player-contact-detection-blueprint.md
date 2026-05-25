1. Problem Understanding

Task type

* Binary temporal event detection from synchronized multi-view sports videos
* Predict whether:
    * player-player (PP) contact occurs
    * player-ground (PG) contact occurs
* Multi-modal learning problem:
    * endzone video
    * sideline video
    * player tracking metadata
* Strong temporal dependency across neighboring frames

Evaluation metric

* Matthews Correlation Coefficient (MCC)
* Threshold-sensitive binary classification metric
* Temporal consistency strongly affects score

Key challenges

* Extremely imbalanced dataset
* Contact events are sparse
* Temporal localization noise:
    * estimated frame may not align perfectly
* Multi-view synchronization
* Small visual target regions
* High GPU memory usage due to video sequences
* Temporal smoothness needed for stable predictions

⸻

2. Data Pipeline (Code-Oriented)

load_data()

Responsibilities

* Load:
    * train_labels.csv
    * tracking metadata
    * helmet detections
    * sample submission
    * extracted video frames
* Build unified dataframe keyed by:
    * game_play
    * frame
    * player ids

Outputs

* train_df
* test_df
* metadata dictionaries
* frame path index

⸻

preprocess()

Core preprocessing logic

1. Synchronize frame timestamps

* Match tracking timestamps to video frames
* Estimate nearest video frame for each sample

2. Merge helmet detections

* Merge bounding boxes for:
    * player 1
    * player 2

3. Generate contact center

For PP:

* midpoint between helmets

For PG:

* player helmet center only

4. Normalize tracking coordinates

* Normalize x/y positions into image-space simulation canvas

5. Create temporal neighbor indices

Store neighboring frames:

* PP:

[-44,-37,-30,-24,-18,-13,-8,-4,-2,0,2,4,8,13,18,24,30,37]

* PG:

[-54,-48,-42,-36,-30,-24,-18,-13,-8,-4,-2,0,2,4,8,13,18,24,30,36,42,48,54]

⸻

feature_engineering()

CNN input features

PP model

Generate:

* endzone sequence
* sideline sequence
* simulated tracking images

Tracking image simulation

Create black canvas:

* draw all players using cv2.circle
* larger bright circles for target pair
* smaller dim circles for background players
* different colors per team

Head masking

Draw black/white circles over:

* target helmets
* forces network attention

Dynamic crop

Crop region:

* centered on interaction
* crop size:

10 × mean helmet size

PG model

* Same video generation
* No tracking simulation images

⸻

split_folds()

Cross-validation strategy

Use:

* GroupKFold
    or
* StratifiedGroupKFold

Grouping:

* game_play

Reason:

* avoid leakage across same game

Typical setup:

* 4 or 5 folds

⸻

3. Model Design

build_model()

Main architecture

3D CNN action recognition network

Recommended backbone:

* ResNet50-irCSN
* imported from:
    * mmaction2
    * timm/video implementation
    * torchvision video models

Why 3D CNN

Task behaves like:

* temporal action classification
    rather than:
* static image classification

⸻

Input structure

PP input

3 branches:

1. Endzone clip
2. Sideline clip
3. Tracking simulation clip

Fusion:

* concatenate along channel dimension
    or
* separate encoders + concat embeddings

PG input

2 branches:

1. Endzone clip
2. Sideline clip

⸻

Pretrained weights

Use:

* Kinetics-400 pretrained weights

Fine-tuning:

* replace classification head
* binary output

⸻

Output head

Typical head:

nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(in_features, 1)
)

⸻

4. Training Strategy

train_one_fold()

Loss function

Binary classification:

BCEWithLogitsLoss()

Optional:

* focal loss for imbalance

⸻

Optimizer

Typical:

AdamW(
    lr=1e-4,
    weight_decay=1e-5
)

⸻

LR scheduler

Linear scheduler:

get_linear_schedule_with_warmup()

⸻

Augmentations

Albumentations pipeline

RandomResizedCrop
RandomGamma
RandomBrightnessContrast
ColorJitter
HueSaturationValue
CLAHE
HorizontalFlip
ShiftScaleRotate
Cutout

Tracking augmentations

Only:

* horizontal flip
* vertical flip

⸻

Training tricks

Mixed precision

Use:

torch.cuda.amp

Gradient clipping

clip_grad_norm_

Random camera swap

Randomly exchange:

* sideline
* endzone

Improves robustness

⸻

Epochs

Very short training:

* 1 epoch full fine-tuning

Reason:

* pretrained action models converge quickly
* overfitting risk high

⸻

Multi-seed training

Train:

* 4 seeds

Average predictions

⸻

5. Validation Strategy

Cross-validation logic

Group-aware folds

Prevent same gameplay leakage.

Separate validation

Train:

* PP model independently
* PG model independently

⸻

OOF generation

Store:

* raw logits
* probabilities
* fold predictions

Use OOF predictions for:

* XGBoost post-processing

⸻

Threshold tuning

Tune threshold:

* per contact type
* using MCC on OOF

⸻

6. Inference Pipeline

predict()

Steps

1. Load trained fold models
2. Generate temporal clips
3. Run AMP inference
4. Average fold predictions
5. Average seed predictions

⸻

TTA

Recommended TTA:

* horizontal flip
* multi-crop

Average logits.

⸻

Ensemble

CNN + pre-XGB ensemble

PP:

prob = 0.2 * pre_xgb + 0.8 * cnn

PG:

prob = 0.15 * pre_xgb + 0.85 * cnn

⸻

post_process()

PP postprocessing

Build temporal features:

[
 prob(t-10),
 ...
 prob(t),
 ...
 prob(t+9)
]

Train XGBoost:

* binary classifier
* temporal smoothing

⸻

PG postprocessing

Features:

1. Ensemble probabilities:

prob(t-15:t+14)

2. CNN probabilities:

cnn_prob(t-10:t+9)

3. Pre-XGB probabilities:

xgb_prob(t-10:t+9)

Train second-stage XGBoost.

⸻

7. Key Tricks (ACTIONABLE)

Most important

1. Temporal frame sampling

Dense sampling near target frame:

* improves temporal localization

2. Head masking

Explicitly highlight target players.

Implementation:

cv2.circle(img, center, radius, color, -1)

3. Tracking simulation images

Convert tracking coordinates into pseudo-images.

4. Separate PP and PG models

Different temporal behaviors.

⸻

Secondary improvements

5. Temporal XGBoost smoothing

Use neighboring probabilities as features.

6. Action-recognition pretrained backbone

Kinetics pretraining crucial.

7. Dynamic interaction-centered crop

Avoid irrelevant field regions.

⸻

Minor tricks

8. Random endzone/sideline swap

9. Multi-seed averaging

10. CLAHE augmentation

11. One-epoch fine-tuning

⸻

8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

import os
import cv2
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from sklearn.model_selection import GroupKFold
from sklearn.metrics import matthews_corrcoef
import xgboost as xgb
# =========================================================
# CONFIG
# =========================================================
class CFG:
    seed = 42
    num_folds = 5
    batch_size = 4
    num_workers = 4
    image_size = 224
    lr = 1e-4
    weight_decay = 1e-5
    epochs = 1
    device = "cuda"
# =========================================================
# UTILS
# =========================================================
def seed_everything(seed):
    pass
# =========================================================
# DATA LOADING
# =========================================================
def load_data():
    """
    Load labels, tracking data,
    helmet detections, frame paths.
    """
    pass
# =========================================================
# PREPROCESSING
# =========================================================
def preprocess(df):
    """
    Merge metadata, align timestamps,
    generate neighboring frame indices.
    """
    pass
# =========================================================
# FEATURE ENGINEERING
# =========================================================
def create_tracking_image(tracking_df):
    """
    Simulate bird-eye-view tracking image.
    """
    pass
def crop_interaction_region(frame, bbox1, bbox2):
    """
    Crop interaction-centered region.
    """
    pass
def mask_target_heads(img, centers):
    """
    Draw black/white circles over helmets.
    """
    pass
# =========================================================
# DATASET
# =========================================================
class ContactDataset(Dataset):
    def __init__(self, df, mode="train", task="PP"):
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        """
        Return:
        - endzone clip
        - sideline clip
        - tracking clip (PP only)
        - label
        """
        pass
# =========================================================
# AUGMENTATIONS
# =========================================================
def build_transforms():
    """
    Albumentations augmentations.
    """
    pass
# =========================================================
# MODEL
# =========================================================
class ContactModel(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Load pretrained irCSN backbone.
        Replace classification head.
        """
    def forward(self, x):
        pass
def build_model():
    pass
# =========================================================
# TRAINING
# =========================================================
def train_one_epoch(model, loader, optimizer, scheduler):
    pass
def valid_one_epoch(model, loader):
    pass
def train_one_fold(fold, train_df, valid_df, task="PP"):
    """
    Train single fold model.
    Save OOF predictions.
    """
    pass
# =========================================================
# VALIDATION
# =========================================================
def search_best_threshold(y_true, y_pred):
    """
    Optimize MCC threshold.
    """
    pass
# =========================================================
# POSTPROCESSING
# =========================================================
def build_temporal_features(df, window=10):
    """
    Create neighboring probability features
    for XGBoost.
    """
    pass
def train_postprocess_xgb(train_features, train_labels):
    pass
# =========================================================
# INFERENCE
# =========================================================
def inference(models, test_df, task="PP"):
    """
    Fold averaging + seed averaging.
    """
    pass
def post_process(pred_df, xgb_model):
    pass
# =========================================================
# FOLD CREATION
# =========================================================
def create_folds(df):
    gkf = GroupKFold(n_splits=CFG.num_folds)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        gkf.split(df, groups=df["game_play"])
    ):
        df.loc[val_idx, "fold"] = fold
    return df
# =========================================================
# MAIN
# =========================================================
def main():
    seed_everything(CFG.seed)
    train_df, test_df = load_data()
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    train_df = create_folds(train_df)
    pp_models = []
    pg_models = []
    # -----------------------------
    # TRAIN PP MODELS
    # -----------------------------
    for fold in range(CFG.num_folds):
        tr_df = train_df[
            (train_df.fold != fold) &
            (train_df.contact_type == "PP")
        ]
        va_df = train_df[
            (train_df.fold == fold) &
            (train_df.contact_type == "PP")
        ]
        model = train_one_fold(
            fold,
            tr_df,
            va_df,
            task="PP"
        )
        pp_models.append(model)
    # -----------------------------
    # TRAIN PG MODELS
    # -----------------------------
    for fold in range(CFG.num_folds):
        tr_df = train_df[
            (train_df.fold != fold) &
            (train_df.contact_type == "PG")
        ]
        va_df = train_df[
            (train_df.fold == fold) &
            (train_df.contact_type == "PG")
        ]
        model = train_one_fold(
            fold,
            tr_df,
            va_df,
            task="PG"
        )
        pg_models.append(model)
    # -----------------------------
    # INFERENCE
    # -----------------------------
    pp_preds = inference(pp_models, test_df, task="PP")
    pg_preds = inference(pg_models, test_df, task="PG")
    # -----------------------------
    # POSTPROCESSING
    # -----------------------------
    pp_final = post_process(pp_preds)
    pg_final = post_process(pg_preds)
    # -----------------------------
    # SAVE SUBMISSION
    # -----------------------------
    submission = pd.concat([pp_final, pg_final])
    submission.to_csv("submission.csv", index=False)
if __name__ == "__main__":
    main()

⸻

Function explanations

Function	Purpose
load_data	Load metadata, labels, frames
preprocess	Synchronize timestamps and create temporal indices
create_tracking_image	Convert tracking data into image
crop_interaction_region	Interaction-focused crop
mask_target_heads	Highlight target helmets
ContactDataset	Generate temporal clips
build_transforms	Albumentations augmentations
ContactModel	3D CNN classifier
train_one_fold	Fold training loop
build_temporal_features	Temporal smoothing features
train_postprocess_xgb	Train second-stage XGB
inference	Fold + seed ensemble
post_process	Final temporal correction

⸻

9. Strategy Priority (IMPORTANT)

1. Most impactful techniques

1. 3D action-recognition backbone (irCSN)
2. Temporal frame sampling
3. Separate PP / PG models
4. Tracking-image simulation
5. Temporal XGBoost postprocessing
6. Interaction-centered crop
7. Kinetics pretrained weights

⸻

2. Secondary improvements

1. Head masking
2. Multi-seed ensemble
3. TTA flips
4. Dynamic temporal windows
5. Threshold optimization with MCC

⸻

3. Minor tricks

1. CLAHE augmentation
2. Random sideline/endzone swapping
3. One-epoch fine-tuning
4. Cutout augmentation
5. Gradient clipping