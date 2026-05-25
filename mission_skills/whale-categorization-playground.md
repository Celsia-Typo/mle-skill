Here is the structured, actionable blueprint for the Whale Classification Siamese Network. This translates the conceptual strategy into a direct implementation plan for a single-file Python script.

## 1. Problem Understanding
* **Task Type:** Fine-grained Image Classification / Open-set Identification (Pairwise Matching).
* **Evaluation Metric:** Mean Average Precision @ 5 (MAP@5).
* **Key Challenges:** Extreme class imbalance (many single-image whales), visual similarity between distinct entities, varied image domains (color vs. grayscale), and varied spatial alignments (flipped flukes, uncentered subjects).

## 2. Data Pipeline (Code-Oriented)
* **`load_data()`**:
    * Load `train.csv` and `sample_submission.csv`.
    * Load pre-computed bounding boxes and the hardcoded `rotate.txt` list.
    * Compute or load Perceptual Hashes (`phash`) for all images. Group images with a Hamming distance $\le 6$, identical sizes, and normalized MSE $< 0.1$.
    * Create an `h2p` (hash-to-picture) mapping that selects the highest-resolution image for each unique hash to filter out inferior duplicates.
* **`preprocess(image_id, augment=False)`**:
    * **Rotation:** Check if `image_id` is in the known flipped list; if so, rotate 180 degrees.
    * **Grayscale:** Convert all images to 1-channel ('L' mode) to equalize colored and B&W domains.
    * **Cropping:** Extract the bounding box, adding a 5% margin (`crop_margin = 0.05`). Adjust the box to enforce a 2.15 width-to-height anisotropy ratio.
    * **Affine Augmentation (if train):** Apply a random transformation matrix combining rotation (-5 to 5 deg), shear (-5 to 5 deg), zoom (0.8 to 1.0), and shift (-5% to 5%).
    * **Normalization:** Resize to 384x384x1. Apply pixel-level normalization (subtract mean, divide by standard deviation per image).
* **`feature_engineering()` (Pair Generator):**
    * Construct a dynamic epoch generator.
    * Ensure each training image appears exactly 4 times per epoch: twice in a matching pair (A, B) and twice in a non-matching pair (A, C).
    * Maintain a strict 50% positive (match) and 50% negative (non-match) class balance in every batch.
* **`split_folds()`**:
    * Group by Whale ID to ensure distinct individuals are kept in their respective folds (GroupKFold), preventing data leakage of specific whales across the train/validation boundary.

## 3. Model Design
* **`build_branch_model()`**:
    * Construct a custom memory-efficient ResNet CNN accepting 384x384x1 inputs.
    * *Block 1:* 384x384 $\rightarrow$ Stride 2 conv + 2x2 Max Pooling (aggressive downsampling to save VRAM).
    * *Block 2:* 96x96 $\rightarrow$ Two VGG-style 3x3 convs.
    * *Blocks 3-6:* Standard ResNet bottleneck architectures (1x1 down, 3x3, 1x1 up + bypass connection). 4 subblocks per spatial resolution.
    * Final layer: Global Max Pooling (to handle uncentered flukes).
* **`build_head_model()`**:
    * Accept two feature vectors, $x$ and $y$, from the branch model.
    * Compute four interaction tensors: $x+y$, $xy$, $|x-y|$, and $(x-y)^2$.
    * Concatenate and pass through a small, shared dense network.
* **`build_siamese_model()`**:
    * Instantiate one Branch Model. Pass Image A and Image B through it to get $x$ and $y$.
    * Pass $x$ and $y$ to the Head Model.
    * Output a single sigmoid probability predicting if A and B match.

## 4. Training Strategy
* **`train_one_fold()`**:
    * **Loss Function:** Binary Cross-Entropy (BCE) for the pairwise matching probability.
    * **Optimizer:** Adam with a learning rate scheduler (e.g., ReduceLROnPlateau).
    * **Hard Negative Mining:** As training progresses, dynamically update the data generator's negative pairs. Run current model inference on random negative pairs and select the ones the model scores highest (most confused) to use in the next epoch.
    * **Tricks:** Enable Automatic Mixed Precision (AMP) to handle the heavy VRAM load on 384x384 images.

## 5. Validation Strategy
* **`validate()`**:
    * Use a holdout set of pairs with the same 50/50 distribution.
    * Additionally, simulate the leaderboard task: compare validation images against the training set corpus, rank the top 5 matches, and calculate the MAP@5 metric to ensure correlation with the competition metric.

## 6. Inference Pipeline
* **`predict()`**:
    * For each test image, extract its feature vector $x$ using the trained Branch Model.
    * Compare $x$ against all (or a filtered subset of) pre-computed feature vectors $y$ from the training set using the Head Model.
* **`post_process()`**:
    * Sort the training images by matching probability for each test image.
    * Extract the corresponding Whale IDs. Take the top 5 unique IDs.
    * Insert a `new_whale` prediction if the top probabilities fall below a tuned threshold.

## 7. Key Tricks (ACTIONABLE)
* **If generating batches $\rightarrow$** Enforce 50% match / 50% mismatch. If epoch > 5, inject hard negative mining by selecting distinct whales that have high cosine similarity in the feature space.
* **If color channels differ $\rightarrow$** Cast everything to grayscale.
* **If comparing feature vectors $\rightarrow$** Do not use L1/L2 distance. Construct the explicit 4-part interaction metric: $x+y$, $xy$, $|x-y|$, $(x-y)^2$.
* **If setting up the CNN $\rightarrow$** Use an aggressive early max-pool to keep VRAM usage manageable, but follow it with a VGG-style block to preserve channel depth before entering ResNet blocks.

## 8. FINAL SINGLE-FILE CODE STRUCTURE

```python
import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# Import other necessary libraries (PIL, imagehash, scipy, etc.)

def seed_everything(seed=42):
    """Locks random seeds for reproducibility across numpy, python, and torch."""
    pass

def load_data(data_dir):
    """
    Reads train.csv, sample_submission.csv.
    Generates/loads perceptual hashes to create the high-res h2p deduplication dict.
    Loads bounding box coordinates.
    Returns clean dataframe and auxiliary dicts.
    """
    pass

def preprocess(image_path, bbox, augment=False):
    """
    Reads image, applies 180-deg rotation if in known list.
    Converts to grayscale ('L').
    Crops using bbox with 0.05 margin and 2.15 anisotropy.
    If augment=True, applies affine transforms (zoom, shift, rotate, shear).
    Resizes to 384x384 and standardizes pixels.
    Returns normalized numpy array / torch tensor.
    """
    pass

def get_pair_generator(df, mode='train', hard_negatives=None):
    """
    Constructs the epoch sequence.
    Ensures each image is used 4 times (2 positive pairs, 2 negative pairs).
    Integrates hard_negatives if provided for adversarial-style training.
    """
    pass

class WhaleDataset(Dataset):
    """PyTorch Dataset wrapping the pair generator and preprocess steps."""
    def __init__(self, pairs, bboxes, augment=False):
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass

def create_folds(df, n_splits=5):
    """Applies GroupKFold on Whale_ID to split data safely."""
    pass

class BranchModel(nn.Module):
    """
    Custom 6-block CNN.
    Block 1: Stride 2 conv + 2x2 MaxPool.
    Block 2: 2x 3x3 Convs.
    Blocks 3-6: ResNet Bottlenecks.
    Final: Global Max Pooling.
    Returns feature vector.
    """
    pass

class HeadModel(nn.Module):
    """
    Takes feature vectors x and y.
    Computes interactions: x+y, x*y, |x-y|, (x-y)^2.
    Passes concatenated result through shared dense layers to a Sigmoid output.
    """
    pass

class SiameseNet(nn.Module):
    """Wraps BranchModel and HeadModel into a single trainable architecture."""
    def __init__(self):
        super().__init__()
        self.branch = BranchModel()
        self.head = HeadModel()
    def forward(self, img1, img2):
        pass

def train_one_fold(fold, train_loader, val_loader, model, criterion, optimizer, scaler):
    """
    Executes training loop with Automatic Mixed Precision (AMP).
    Tracks running BCE loss and binary accuracy.
    Optionally updates hard negative mining queue.
    """
    pass

def validate(model, val_loader):
    """
    Evaluates model on validation pairs.
    Simulates leaderboard ranking to output MAP@5 estimate.
    """
    pass

def inference(models, test_df, train_features):
    """
    Extracts test features.
    Compares against training features using the Head Model.
    Averages predictions across fold models.
    Outputs top 5 predictions + new_whale logic.
    """
    pass

def main():
    seed_everything(42)
    df, bboxes, h2p = load_data('./input')
    folds = create_folds(df)
    
    models = []
    for fold in range(5):
        # Setup data loaders
        # Initialize SiameseNet, AdamW, BCEWithLogitsLoss, GradScaler
        # model = train_one_fold(...)
        # validate(model, ...)
        # models.append(model)
        pass
        
    # preds = inference(models, test_df, ...)
    # save_submission(preds)

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority

1.  **Most Impactful Techniques:** * The 50/50 matching/non-matching pair generation with strict image repetition constraints (4 times per epoch).
    * Hard negative mining for distinct whales (adversarial-inspired pairs).
    * The custom 4-part vector interaction Head Model ($x+y$, $xy$, $|x-y|$, $(x-y)^2$) instead of standard L1/L2 distance.
2.  **Secondary Improvements:**
    * Grayscale conversion to harmonize domains.
    * Perceptual hashing deduplication to ensure the network trains on the highest resolution variant of a duplicated fluke.
    * Precise affine cropping using bounding boxes with a 0.05 margin.
3.  **Minor Tricks:**
    * Manual 180-degree rotation of known inverted flukes.
    * Aggressive early max-pooling in the CNN branch to prevent VRAM Out-Of-Memory errors on 384x384 inputs.