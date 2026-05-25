## 1. Problem Understanding
* **Task type:** Fine-Grained Visual Categorization (FGVC) / Multi-class Image Classification.
* **Evaluation metric:** Macro F1-Score / Accuracy (Standard for classification leaderboards).
* **Key challenges:** Extreme multi-class setting (thousands of species), long-tailed class distribution, high intra-class variance, and high inter-class similarity. Hardware memory constraints when training large Transformer models.

## 2. Data Pipeline (Code-Oriented)
* `load_data()`: Parse the competition metadata (usually JSON or CSV). Extract image file paths and hierarchical labels: `family`, `genus`, and `species`.
* `preprocess()`: Map string labels to contiguous integer IDs using `LabelEncoder`. Ensure all target variables (family, genus, species) have their own distinct label mappings.
* `feature_engineering()`: Add a frequency column for the `species` target to calculate dynamic margins for ArcFace later.
* `split_folds()`: Implement `StratifiedKFold` (5 folds) using the `species` label to ensure the long-tailed distribution is respected across all validation sets. 
* **Dataset Class (`torch.utils.data.Dataset`)**: 
    * Implement standard Swin-Transformer data augmentations using `albumentations` or `torchvision.transforms` (RandAugment, MixUp, CutMix, RandomResizedCrop).
    * Implement square resizing functionality.

## 3. Model Design
* `build_model(config)`: Utilize the `timm` (PyTorch Image Models) library.
* **Model Types:** Primary backbone is `swinv2_base_window12_192_22k` (scaled to 384x384). Secondary backbones for ensemble include ConvNeXt-B, DeiT-III, ResNeSt-101, EfficientNet-B6, CSwin-L, and Swin-L.
* **Pretrained Usage:** Strictly initialize with `ImageNet22k` weights.
* **Custom Head Architecture:**
    * Remove the default classifier head.
    * *Branch 1 (Species - Main):* Pass features through a Subcenter-ArcFace module (k=3 centers per class) with dynamically calculated margins (inverse to class frequency).
    * *Branch 2 (Species - Aux):* Pass features through a standard Fully Connected (Linear) layer.
    * *Branch 3 (Genus):* Linear layer predicting the genus.
    * *Branch 4 (Family):* Linear layer predicting the family.
    * *Forward pass output:* Return logits for ArcFace, Aux Species, Genus, and Family.

## 4. Training Strategy
* `train_one_fold()`: Implement a two-phase training loop to handle VRAM limits.
    * **Phase 1 (Epochs 0-X):** Freeze the backbone parameters. Train only the custom multi-head using a large batch size (`BS=512`).
    * **Phase 2 (Epochs X-100):** Unfreeze the backbone. Reduce the batch size (`BS=256`). Train full network parameters.
* **Loss Function:** A combined weighted loss.
    * `Loss = ArcFace_Loss(Species) + CE_Loss(Aux_Species) + CE_Loss(Genus) + CE_Loss(Family)`
* **Optimizer:** `AdamW` with an initial learning rate of `5e-4`. Use Cosine Annealing learning rate scheduler.
* **Tricks:** * Use `torch.cuda.amp.autocast()` and `GradScaler` for mixed-precision training to save memory.
    * Gradient clipping is necessary when training Transformers.

## 5. Validation Strategy
* **Cross-validation logic:** Evaluate models at the end of each epoch using the out-of-fold (OOF) dataset.
* **Validation Data Augmentation:** Implement the "5-crop" validation technique. Resize the image to multiple resolutions (e.g., 400, 416, 448, 480, 512), apply a center crop of 384x384 to each, pass all 5 crops through the model, and average the predicted probabilities.
* **OOF Generation:** Save OOF probabilities and calculate the final metric to verify local CV correlates with the public LB.

## 6. Inference Pipeline
* `predict()`: Load saved weights for the entire ensemble array.
* **TTA / Ensemble:** * Apply the same 5-crop Test Time Augmentation (TTA) used in validation.
    * For the final species prediction, combine Branch 1 and Branch 2: `Final_Prob = Softmax(ArcFace_Logits) + Softmax(Aux_Species_Logits)`.
    * Calculate a weighted average of probabilities across all ensemble backbones. Use their Public LB scores as the scaling weights.
* **Post_process():** Apply a prior-probability shift. Adjust the raw predicted probabilities slightly based on the training set class frequencies (a standard Kaggle trick for long-tailed distributions to gain the mentioned 0.001 boost).

## 7. Key Tricks (ACTIONABLE)
* **If defining the ArcFace module** → Use *Subcenter* ArcFace. Set `k` (subcenters) to 2 or 3 to allow the model to learn multiple visual clusters for the same highly-variant species.
* **If configuring ArcFace margins** → Calculate the frequency of each class in the training data. Assign smaller margins to frequent classes and larger margins to rare classes.
* **If building the optimizer** → Set LR exactly to `5e-4` (up from baseline `2e-4`).
* **If handling input resolution** → Use square resizing. Train at 384x384. Test/Crop from resolutions mathematically larger than 384.
* **If ensembling** → Do NOT use simple averaging. Weight model `i`'s predictions by `LB_Score_i / Sum(All_LB_Scores)`.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# ... other standard imports (timm, sklearn, albumentations) ...

def seed_everything(seed):
    # Lock random states for reproducibility across numpy, torch, python random
    pass

def load_data():
    # Read CSV files, map taxonomy strings to integer IDs, return DataFrames
    pass

def preprocess(df):
    # Calculate class frequencies for dynamic margins, handle missing data
    pass

def split_folds(df):
    # Apply StratifiedKFold based on the 'species' column
    pass

class HerbariumDataset(Dataset):
    # Implement __init__, __len__, and __getitem__
    # Return image tensors and a dictionary of labels (species, genus, family)
    # Apply standard Swin augmentations for train, and multi-crop for valid/test
    pass

class SubcenterArcFace(nn.Module):
    # Implement Subcenter ArcFace logic with dynamic margin calculation
    pass

class KaggleModel(nn.Module):
    # Initialize timm backbone (e.g., swinv2_base) with ImageNet22k weights
    # Strip original classifier
    # Define SubcenterArcFace head, Aux Linear head, Genus head, Family head
    # Return all logits in forward()
    pass

def train_one_epoch(model, dataloader, optimizer, scaler, phase):
    # Handle AMP loop, compute combined multi-level loss, backward pass
    pass

def validate(model, dataloader):
    # Apply 5-crop TTA logic internally, compute OOF metrics
    pass

def train_one_fold(fold, train_df, val_df, config):
    # Manage Two-Phase training (Freeze backbone -> Train -> Unfreeze -> Train)
    # Manage LR scheduler and model checkpointing
    pass

def inference(models, test_loader):
    # Loop through models, apply weighted averaging based on LB scores
    # Combine ArcFace and Aux Softmax outputs
    # Apply post-processing frequency shift
    pass

def main():
    # 1. Pipeline execution
    train_df, test_df = load_data()
    train_df = preprocess(train_df)
    train_df = split_folds(train_df)
    
    # 2. Training
    models = []
    for fold in range(5):
        model = train_one_fold(fold, train_df, train_df[train_df.fold == fold], config)
        models.append(model)
        
    # 3. Inference & Submission
    preds = inference(models, test_df)
    # format and save submission.csv
    pass

if __name__ == "__main__":
    # Config dict
    main()
```

## 9. Strategy Priority (IMPORTANT)

1.  **Most impactful techniques:** Subcenter-ArcFace with dynamic margins combined with Multi-level hierarchical loss (Species + Genus + Family branches).
2.  **Secondary improvements:** Multi-resolution 5-crop Test Time Augmentation (TTA) and weighted model ensembling using public LB scores.
3.  **Minor tricks:** Progressive unfreezing to bypass batch-size memory constraints and post-processing probability shifts.