## 1. Problem Understanding
* **Task type:** Fine-Grained Visual Categorization (FGVC) / Multi-class Image Classification.
* **Evaluation metric:** Macro F1-Score (standard for highly imbalanced FGVC tasks).
* **Key challenges:** * Extreme class imbalance (long-tailed distribution).
    * Massive number of target classes (32,093 species).
    * Subtle visual differences between categories requiring both local and global feature extraction.

## 2. Data Pipeline (Code-Oriented)
* **`load_data()`:** Parse training CSV/JSON to map image paths to labels. Identify the top 5,000 most frequent categories to create a subset dataframe for Stage 1 training, and keep the full dataset for Stage 2.
* **`preprocess()`:** Implement Albumentations/Torchvision transforms pipeline.
    * Training: `ResizeShorterSide` (dynamic between `size+32` and `size+128`), `RandomCrop` to target `size` (320, 380, 448, or 456 depending on backbone), `RandAugment` (2 operations, dynamic magnitude 1-11), `RandomErasing`, normalization.
    * Validation/Inference: Resize, CenterCrop, normalization.
* **`feature_engineering()`:** Create custom PyTorch `Sampler` classes for balanced sampling. The sampler must guarantee a fixed number of images per category per epoch (75 for Stage 1; 25 for Stage 2).
* **`split_folds()`:** Use `StratifiedKFold` to ensure rare classes are distributed across folds properly, though Stage 1 only trains on the frequent subset.

## 3. Model Design
* **`build_model()`:** Construct a custom `nn.Module` using `timm` for backbones (EfficientNet-B3/4/5-noisy-student, InceptionV4, SEResNeXt50).
* **Architecture specifics:**
    * **Local Branch:** Extract intermediate feature maps (blocks 4, 5, and 6). Pass each through a custom One-Squeeze Multi-Excitation (OSME) block. Apply Global Average Pooling (GAP) and concatenate them into a single local feature vector. Add a linear classifier layer.
    * **Global Branch:** Take the final "head" block features. Pass through a BNNeck (Batch Normalization layer) and an ArcFace margin layer (scale `s=16`, margin `m=0.1`).
    * **Inference Output:** Average the logits from three sources: the local classifier, the BNNeck feature classifier, and the ArcFace classifier.

## 4. Training Strategy
* **`train_stage_one()`:** Train the model using only the subset of the top 5,000 frequent classes. Run for 75 epochs using the 75-samples-per-class sampler.
* **`weight_imprinting()`:** Before Stage 2, extract features for the remaining 27,093 rare categories using the Stage 1 backbone. Compute the mean L2-normalized feature vector for each rare class and use these to initialize the weights of the fully connected classification layer for the rare classes.
* **`train_stage_two()`:** Unfreeze/retrain the whole network on all 32,093 classes for 105 epochs. Use the 25-samples-per-class sampler.
* **Optimizer/Scheduler:** `AdamW` optimizer. Cosine Annealing learning rate scheduler with a linear warmup phase.
* **Loss Function:** A combined loss function summing standard Cross-Entropy (without label smoothing), Entropy Loss, and ArcFace Loss.
* **Tricks:** Use PyTorch AMP (Automatic Mixed Precision) and Gradient Accumulation to achieve an effective batch size of 2048.

## 5. Validation Strategy
* **Cross-validation logic:** Evaluate on the local validation fold using Macro F1-score. 
* **OOF generation:** Save Out-Of-Fold (OOF) logits (not just probabilities) for every validation sample to be used later by the 2nd-level stacking model.

## 6. Inference Pipeline
* **`predict()`:** Run test images through all trained single models to extract logit predictions.
* **Ensemble:** Pass the concatenated logits of shape `(batch, num_models, num_classes, 1)` into the custom `StackingCNN`. The stacking network uses 1D/2D convolutions over the model dimension to produce final blended logits.
* **`post_process()`:** Group test predictions by the predicted species (top-1 confidence). Sort each group by confidence descending. If a prediction is ranked 11th or lower for that specific species, overwrite its prediction with its top-2 category. (Exploiting the test set constraint of max 10 samples per species).

## 7. Key Tricks (ACTIONABLE)
* **If handling extreme long-tail distributions** → Implement a two-stage approach with weight imprinting. Do not start training from scratch on rare classes.
* **If aiming for large batch sizes (2048) on limited VRAM** → Accumulate gradients over multiple steps (`loss.backward()` without `optimizer.step()`) and explicitly wrap training blocks in `torch.cuda.amp.autocast()`.
* **If test dataset has strict count constraints per class** → Implement deterministic rank-based post-processing to reassign classes when limits are exceeded.
* **Hyperparameters:** ArcFace scale = 16, margin = 0.1. Batch size = 2048. RandAugment ops = 2, magnitude = random(1, 11).

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from collections import OrderedDict

# --- CONFIGURATION ---
class Config:
    seed = 42
    img_size = 320
    batch_size = 64
    accumulate_steps = 32  # To hit 2048 effective batch size
    stage1_epochs = 75
    stage2_epochs = 105
    top_k_frequent = 5000
    total_classes = 32093
    # Add other hyperparameters...

def seed_everything(seed):
    # Set seeds for torch, numpy, random, and cudnn determinism
    pass

# --- DATA MANIPULATION ---
def load_data(csv_path):
    # Read CSV, compute class frequencies, identify top 5000 classes
    # Return train_df, test_df
    pass

class HerbariumDataset(Dataset):
    # Implements image loading and Albumentations augmentations
    pass

class BalancedSampler(Sampler):
    # Yields exactly `n` samples per category per epoch
    pass

def create_dataloaders(df, stage, config):
    # Build datasets, samplers, and dataloaders based on stage 1 or 2
    pass

# --- MODEL ARCHITECTURE ---
class OSMEBlock(nn.Module):
    # One-Squeeze Multi-Excitation block definition
    pass

class ArcFaceMarginProduct(nn.Module):
    # ArcFace layer (s=16, m=0.1)
    pass

class CustomHerbariumModel(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        # 1. Load timm backbone
        # 2. Extract specific blocks (4, 5, 6)
        # 3. Setup OSME blocks for local features
        # 4. Setup BNNeck + ArcFace for global features
        pass

    def forward(self, x, labels=None):
        # Pass through backbone
        # Extract local features -> OSME -> concat -> local logits
        # Extract global features -> BNNeck -> ArcFace -> global logits
        # Return losses during training, or averaged logits during eval
        pass

class StackingCNN(nn.Module):
    # Implements the 2-layer CNN for ensembling logits
    def __init__(self, num_models, num_channels=128):
        super().__init__()
        # Conv layers over the model dimension
        pass
    def forward(self, logits):
        pass

# --- TRAINING PIPELINE ---
def combined_loss(logits_dict, labels):
    # Sums CrossEntropy, Entropy Loss, and ArcFace Loss
    pass

def weight_imprinting(model, dataloader, config):
    # Extract features for rare classes using Stage 1 model
    # Compute L2 normalized mean vectors
    # Directly inject into model's classifier weights
    pass

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, config):
    # Standard PyTorch training loop with AMP and gradient accumulation
    pass

def validate(model, dataloader):
    # Compute validation Macro F1 and extract OOF logits
    pass

def train_pipeline(df, config):
    # 1. Split top 5000 classes for Stage 1
    # 2. Train Stage 1 for 75 epochs
    # 3. Perform weight imprinting for remaining classes
    # 4. Train Stage 2 on all classes for 105 epochs
    pass

def train_stacking_ensemble(oof_logits, labels):
    # Train the 1D/2D Conv stacking model using OOF logits
    pass

# --- INFERENCE & POSTPROCESSING ---
def post_process_predictions(predictions):
    # Group by species
    # Sort by top-1 confidence
    # Cap at 10 predictions per species; downgrade others to top-2
    pass

def inference(models, stacking_model, test_loader):
    # Generate single model logits -> stack -> final logits
    # Convert to predictions and run post_process_predictions()
    pass

def main():
    seed_everything(Config.seed)
    train_df, test_df = load_data('train.csv')
    
    # K-Fold logic here if generating full OOF, else just standard train/val split
    # For blueprint, representing single pipeline
    trained_models = train_pipeline(train_df, Config)
    
    # Run Inference
    # preds = inference(trained_models, StackingCNN_Model, test_loader)
    # save_submission(preds)

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1.  **Most impactful techniques:** Two-stage training with balanced class sampling and weight imprinting (fixes the massive long-tail problem).
2.  **Secondary improvements:** Model architecture modifications (OSME for local features + ArcFace/BNNeck for global features) to separate subtle species differences. Stacking CNN ensemble.
3.  **Minor tricks:** Rule-based post-processing (capping test set predictions to 10 per class based on confidence).