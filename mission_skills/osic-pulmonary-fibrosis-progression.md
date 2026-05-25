1. Problem Understanding

Task type

* Medical tabular + imaging regression
* Predict future lung function progression (FVC) for pulmonary fibrosis patients
* Output:
    * Predicted FVC value
    * Prediction confidence (uncertainty)
* Hybrid modeling:
    * Quantile regression on tabular longitudinal features
    * CNN regression on CT scan slices

Evaluation metric

Competition metric:

* Modified Laplace Log Likelihood
* Requires:
    * Accurate FVC prediction
    * Proper uncertainty/confidence estimation

Implications for implementation:

* Predict both mean target and confidence interval
* Overconfident predictions are heavily penalized
* Stable uncertainty estimation is critical

Key challenges

* Extremely small dataset (~200 patients)
* CT scans are 3D but training commonly uses sampled 2D slices
* Public leaderboard overfitting traps
* Weak correlation between local CV and LB
* High sensitivity to feature leakage and unstable handcrafted features
* Need robust blending between tabular and image models

⸻

2. Data Pipeline (Code-Oriented)

load_data()

Inputs

Load:

* train.csv
* test.csv
* sample_submission.csv
* DICOM CT scan folders

Logic

1. Read patient metadata
2. Read longitudinal patient records
3. Build patient-level dictionary:

patient_id -> list of CT slices

4. Load target:

FVC

5. Extract baseline week per patient

⸻

preprocess()

Tabular preprocessing

* Encode categorical variables:
    * Sex
    * SmokingStatus
* Normalize continuous variables:
    * Age
    * Weeks
    * Baseline FVC

Important decision

DO NOT use:

* Percent feature

Reason:

* Strong public LB overfitting
* Poor private generalization

Longitudinal feature construction

For each patient:

* Baseline week
* Baseline FVC
* Relative week offset:

week_diff = target_week - baseline_week

⸻

feature_engineering()

Quantile regression features

Construct:

* Age
* Sex encoding
* Smoking encoding
* Baseline FVC
* Week delta

Optional:

* Interaction terms:

baseline_fvc * week_delta
age * smoking

CT image preprocessing

For EfficientNet:

1. Sample fixed number of slices per patient
2. Resize:

512 -> 256 or 224

3. HU normalization
4. Convert grayscale to 3-channel image

Avoid:

* Heavy augmentations
* Histogram features
* Lung volume handcrafted features

These were reported ineffective.

⸻

split_folds()

Patient-level CV

Critical:

* Never split rows independently

Use:

GroupKFold

Groups:

Patient

Possible setup:

n_splits = 5

Validation philosophy:

* Emphasize robustness over public LB correlation
* Discard models failing both CV and LB

⸻

3. Model Design

build_model()

Model A — Quantile Regression MLP

Inputs

Tabular engineered features

Architecture

Simple MLP:

Linear
BatchNorm
ReLU
Dropout
Linear
ReLU
Linear

Outputs:

* p20
* p50
* p80 quantiles

Loss

Quantile loss:

pinball_loss

Notes

* Train from scratch
* Main high-weight model in ensemble

⸻

Model B — EfficientNet-B5

Inputs

2D CT slices

Backbone

Use:

timm.create_model('efficientnet_b5')

Output

Regression head:

FVC prediction

Confidence estimation:

* Fixed quantile assumption
* Simpler inference
* Use median quantile = 0.5

Important implementation detail

Avoid expensive quantile search during inference.

Directly:

quantile = 0.5

This greatly speeds up inference.

⸻

Pretrained usage

* ImageNet pretrained EfficientNet optional
* Can also train from scratch

Given tiny dataset:

* Limited fine-tuning epochs preferred

⸻

4. Training Strategy

train_one_fold()

Quantile model

Epochs

600 epochs

Optimizer

AdamW
lr = 1e-3

Scheduler

ReduceLROnPlateau

Loss

Custom quantile loss:

q = [0.2, 0.5, 0.8]

Tricks

* Early stopping
* Gradient clipping
* Small batch size
* Feature normalization

⸻

EfficientNet training

Epochs

30 epochs

Optimizer

Adam
lr = 1e-4

Loss

Regression:

L1Loss

or

SmoothL1Loss

Tricks

* AMP mixed precision
* Freeze early layers initially
* Minimal augmentation

Avoid:

* Aggressive augmentation pipelines

⸻

5. Validation Strategy

Cross-validation logic

Core rule

Patient-level validation:

GroupKFold(groups=Patient)

OOF generation

Store:

* OOF FVC predictions
* OOF confidence predictions

Use OOF to:

* Tune ensemble weights
* Inspect prediction distributions

⸻

Distribution diagnostics

After every fold:
Generate plots:

* Predicted FVC histogram
* Confidence histogram
* Fold-wise prediction scatter

Purpose:

* Detect spikes
* Detect collapsed distributions
* Detect inference bugs

This is considered a major debugging tool.

⸻

6. Inference Pipeline

predict()

Quantile model inference

Generate:

p20, p50, p80

Final:

fvc = p50
confidence = p80 - p20

⸻

EfficientNet inference

Slice aggregation

For each patient:

1. Predict per slice
2. Average predictions

patient_pred = mean(slice_preds)

⸻

Ensemble

Weighted blending:

final_pred =
    0.6 * quantile_model +
    0.4 * efficientnet

Confidence:

* Primarily from quantile model

Observation:

* Quantile model generalized better

⸻

post_process()

Confidence clipping

Prevent extreme confidence:

confidence = np.clip(confidence, 70, 1000)

Optional smoothing

Do NOT remove spikes unless validated.

Author observed:

* Spike removal hurt CV/private LB

⸻

7. Key Tricks (ACTIONABLE)

Most important

Remove Percent feature

DROP_COLUMNS = ['Percent']

Improved private LB stability.

⸻

Use patient-level GroupKFold

GroupKFold(groups=Patient)

Critical for leakage prevention.

⸻

Favor tabular quantile model

Use larger ensemble weight:

0.6~0.7

⸻

Simplify EfficientNet quantile handling

Instead of expensive search:

quantile = 0.5

Huge inference speedup.

⸻

Secondary improvements

Reduce CNN training epochs

30 epochs

Helps avoid overfitting.

⸻

Analyze prediction distributions

Always plot:

* Confidence histograms
* FVC histograms

Useful for bug detection.

⸻

Minimal image augmentation

Avoid strong transforms.

⸻

Things that failed

Avoid wasting implementation time on:

* Lung volume features
* Histogram image features
* Tree models (XGBoost)
* Logistic regression ensembles
* Strong CT augmentations
* Spike removal heuristics

⸻

8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

import os
import gc
import cv2
import timm
import random
import pydicom
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# ====================================================
# CONFIG
# ====================================================
class CFG:
    seed = 42
    n_folds = 5
    img_size = 224
    batch_size = 16
    effnet_epochs = 30
    quantile_epochs = 600
    effnet_lr = 1e-4
    quantile_lr = 1e-3
    quantiles = [0.2, 0.5, 0.8]
# ====================================================
# UTILS
# ====================================================
def seed_everything(seed):
    pass
# ====================================================
# DATA LOADING
# ====================================================
def load_data():
    """
    Load train/test CSVs and DICOM paths.
    """
    pass
# ====================================================
# PREPROCESSING
# ====================================================
def preprocess(df):
    """
    Encode categorical features.
    Remove Percent feature.
    Normalize columns.
    """
    pass
# ====================================================
# FEATURE ENGINEERING
# ====================================================
def feature_engineering(df):
    """
    Build longitudinal features:
    - baseline FVC
    - week delta
    - interactions
    """
    pass
# ====================================================
# FOLD CREATION
# ====================================================
def create_folds(df):
    """
    GroupKFold by patient.
    """
    pass
# ====================================================
# DATASET
# ====================================================
class CTDataset(Dataset):
    def __init__(self, df, mode='train'):
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass
# ====================================================
# TABULAR MODEL
# ====================================================
class QuantileMLP(nn.Module):
    def __init__(self, n_features):
        pass
    def forward(self, x):
        pass
# ====================================================
# IMAGE MODEL
# ====================================================
class EfficientNetModel(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass
# ====================================================
# LOSSES
# ====================================================
def quantile_loss(preds, targets, quantiles):
    pass
# ====================================================
# TRAINING
# ====================================================
def train_quantile_model(fold, train_df, val_df):
    """
    Train tabular quantile regression model.
    """
    pass
def train_effnet_model(fold, train_loader, val_loader):
    """
    Train EfficientNet-B5 model.
    """
    pass
# ====================================================
# VALIDATION
# ====================================================
def validate(model, val_loader):
    """
    Generate OOF predictions.
    """
    pass
def plot_prediction_distributions(preds):
    """
    Debugging visualization.
    """
    pass
# ====================================================
# INFERENCE
# ====================================================
def predict_quantiles(model, test_df):
    pass
def predict_effnet(model, test_loader):
    pass
def ensemble_predictions(q_preds, cnn_preds):
    """
    Weighted blending.
    """
    pass
def post_process(preds, confidence):
    """
    Clip confidence range.
    """
    pass
# ====================================================
# MAIN
# ====================================================
def main():
    seed_everything(CFG.seed)
    train_df, test_df = load_data()
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    train_df = create_folds(train_df)
    quantile_models = []
    effnet_models = []
    for fold in range(CFG.n_folds):
        trn_df = train_df[train_df.fold != fold]
        val_df = train_df[train_df.fold == fold]
        q_model = train_quantile_model(
            fold,
            trn_df,
            val_df
        )
        quantile_models.append(q_model)
        train_loader = ...
        val_loader = ...
        cnn_model = train_effnet_model(
            fold,
            train_loader,
            val_loader
        )
        effnet_models.append(cnn_model)
    q_preds = predict_quantiles(
        quantile_models,
        test_df
    )
    cnn_preds = predict_effnet(
        effnet_models,
        test_loader
    )
    preds, confidence = ensemble_predictions(
        q_preds,
        cnn_preds
    )
    preds, confidence = post_process(
        preds,
        confidence
    )
    save_submission(preds, confidence)
if __name__ == "__main__":
    main()

⸻

Function explanations

load_data()

Reads CSV metadata and DICOM image paths.

preprocess()

Encodes categorical variables and removes unstable features.

feature_engineering()

Builds longitudinal progression features.

create_folds()

Creates patient-level GroupKFold splits.

CTDataset

Loads sampled CT slices dynamically.

QuantileMLP

Tabular uncertainty-aware regression model.

EfficientNetModel

CNN for CT slice regression.

quantile_loss()

Implements pinball loss.

train_quantile_model()

Trains MLP quantile regressor.

train_effnet_model()

Trains EfficientNet-B5.

validate()

Produces OOF predictions.

plot_prediction_distributions()

Visual debugging utility.

ensemble_predictions()

Blends tabular and image predictions.

post_process()

Applies confidence clipping.

⸻

9. Strategy Priority (IMPORTANT)

1. Most impactful techniques

1. Removing Percent feature
2. Patient-level GroupKFold
3. Quantile regression modeling
4. Conservative CNN weighting
5. Proper uncertainty estimation

⸻

2. Secondary improvements

1. Simplified EfficientNet quantile inference
2. Reduced CNN epochs
3. Prediction distribution diagnostics
4. Lightweight preprocessing

⸻

3. Minor tricks

1. Confidence clipping
2. Interaction features
3. Minimal TTA
4. AMP training
5. Slice averaging strategy