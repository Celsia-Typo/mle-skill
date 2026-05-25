Here is the structured, executable solution blueprint to guide the generation of a single Python script for the 1st place Automated Essay Scoring 2.0 solution.

## 1. Problem Understanding
* **Task type:** Ordinal regression / Text sequence classification (predicting integer scores from 1 to 6 based on essay text).
* **Evaluation metric:** Quadratic Weighted Kappa (QWK).
* **Key challenges:** * Significant domain shift and scoring criteria misalignment between the "old" (Persuade corpus) and "new" (competition-specific) datasets.
    * Extreme target imbalance, especially at the tails (very few scores of 1 or 6).
    * High variance across different random seeds, making model selection unreliable without averaging.

## 2. Data Pipeline (Code-Oriented)
* **`load_data()`:** Read the training datasets. Separate the `old_df` (Persuade corpus, ~13k rows) and the `new_df` (competition data, ~4.5k rows).
* **`preprocess(df)`:** Clean text and adjust tokenizer behavior. Specifically, add `\n` and ` ` (space) as special tokens to the tokenizer.
* **`feature_engineering(df)`:** Depending on the loss function chosen, map target integer scores (1-6) to floats. If using Binary Cross Entropy (BCE), scale the targets to a `[0, 1]` range.
* **`split_folds(df)`:** Create a 5-fold cross-validation split using stratified sampling. The stratification target must be a concatenation of `prompt_id` and the `score` label.

## 3. Model Design
* **`build_model(model_name, pooling_type)`:** Initialize a Hugging Face `AutoModelForSequenceClassification` with 1 regression output.
* **Model types:** Primarily `microsoft/deberta-v3-large` (with `deberta-v3-base` optionally used for ensemble diversity).
* **Architecture adjustments:** * Set max context length to 1024.
    * Implement configurable pooling layers (extracting the `CLS` token or using `GeM` pooling).
    * Initialize the classification head weights using a normal distribution (`mean=0.0`, `std=0.02`, `bias=0`).
    * Set Dropout to 0.

## 4. Training Strategy
* **`train_one_fold(fold_id, old_df, new_df)`:** Implement a two-stage training loop.
    * **Stage 1 (Pre-training):** Train the model exclusively on `old_df`. 
    * **Stage 2 (Fine-tuning):** Load the Stage 1 weights and fine-tune exclusively on `new_df`.
* **Loss function:** Mean Squared Error (MSE) or Binary Cross Entropy (BCE) with scaled targets.
* **Optimizer / Params:** * Batch size: 8.
    * Differential Learning Rates (Stage 1): Torso `1e-5`, Head `2e-5`.
    * Differential Learning Rates (Stage 2): Torso `5e-6`, Head `1e-5`.
    * Weight decay: 0.01.
    * Scheduler: Cosine decay with warmup (15% of the pre-training steps).
* **Tricks:** Apply gradient clipping at `10`. Wrap the training logic in a loop to execute 3 distinct random seeds per fold.

## 5. Validation Strategy
* **Cross-validation logic:** Generate out-of-fold (OOF) predictions exclusively from the Stage 2 (fine-tuned on new data) models.
* **OOF generation:** Store the raw float predictions. Average the predictions of the 3 different seeds before calculating the CV score. Calculate the QWK using optimal thresholds (derived in post-processing).

## 6. Inference Pipeline
* **`predict(models, test_df)`:** Run inference on the test set using all saved fold models across all 3 seeds. Average the float outputs.
* **Ensemble logic:** Use a simple average of the model predictions to prevent overfitting the small test set.
* **`post_process(predictions, targets)`:** Implement a float-to-integer thresholding function. 
    * Use `scipy.optimize.minimize` with the "Powell" method.
    * Objective function: Minimize `1 - QWK`.
    * Run the optimization from 15 different random starting arrays and average the resulting threshold bounds.

## 7. Key Tricks (ACTIONABLE)
* **If tackling the data domain shift → do Two-Stage Training:** Always pre-train on the external Persuade dataset, then fine-tune on the specific competition dataset.
* **If model variance is high → do 3-Seed Averaging:** Run training, OOF generation, threshold finding, and test inference across 3 different random seeds and average the results.
* **If utilizing the "old" data for higher performance → do Pseudo Labelling:** Use the Stage 2 fine-tuned model to predict float scores for the `old_df`. Replace the original `old_df` labels with `mean(original_label, predicted_float)`. Retrain the pipeline using these pseudo-labels.
* **If targets are heavily imbalanced → do Custom Thresholding:** Never use standard rounding (e.g., `1.5`, `2.5`). Calculate dataset-specific thresholds by optimizing the QWK metric directly via Scipy.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

# --- CONFIGURATION ---
class Config:
    seed = 42
    n_splits = 5
    model_name = "microsoft/deberta-v3-large"
    max_len = 1024
    batch_size = 8
    epochs = 2
    lr_torso_pre = 1e-5
    lr_head_pre = 2e-5
    lr_torso_fine = 5e-6
    lr_head_fine = 1e-5
    weight_decay = 0.01
    grad_clip = 10
    num_seeds = 3

# --- UTILITIES ---
def seed_everything(seed):
    """Sets random seeds for reproducibility across numpy, torch, and python."""
    pass

def compute_qwk(y_true, y_pred):
    """Calculates Quadratic Weighted Kappa."""
    pass

# --- DATA PIPELINE ---
def load_data():
    """Loads train, test, and external persuade data. Returns old_df, new_df, test_df."""
    pass

def preprocess_text(df, tokenizer):
    """Tokenizes text, adds special tokens (\n, ' '). Returns encoded dataset."""
    pass

def feature_engineering(df):
    """Scales integer labels to [0, 1] for BCE loss. Returns adjusted df."""
    pass

def create_folds(df):
    """Generates 5-fold stratified splits using prompt_id + score. Returns fold array."""
    pass

# --- MODELING ---
def build_model(config):
    """
    Initializes DeBERTa, sets dropout to 0, attaches custom regression head (CLS/GeM), 
    and applies custom weight initialization.
    """
    pass

# --- TRAINING ---
def train_stage(model, dataloader, optimizer, scheduler, is_pretrain=True):
    """Executes one epoch of training. Handles grad clipping and specific loss (MSE/BCE)."""
    pass

def train_one_fold(fold, old_df, new_df, config, seed):
    """
    Executes the 2-stage training strategy:
    1. Pre-train model on old_df.
    2. Fine-tune model on new_df.
    Returns the fine-tuned model and OOF predictions for new_df.
    """
    pass

def apply_pseudo_labels(old_df, trained_models):
    """
    Generates predictions on old_df using fine-tuned models. 
    Updates old_df targets to mean(original, prediction).
    """
    pass

# --- VALIDATION & POST-PROCESSING ---
def optimize_thresholds(y_true, y_pred_float):
    """
    Uses scipy.optimize.minimize (Powell) to find custom thresholds minimizing 1 - QWK.
    Runs 15 random initializations and averages the bounds.
    """
    pass

def apply_thresholds(y_pred_float, thresholds):
    """Converts continuous float predictions to integer classes 1-6 using custom bounds."""
    pass

# --- INFERENCE ---
def inference(models, test_df, thresholds):
    """Runs test data through all models/seeds, averages float outputs, applies thresholds."""
    pass

# --- MAIN EXECUTION ---
def main():
    # 1. Setup & Load
    seed_everything(Config.seed)
    old_df, new_df, test_df = load_data()
    
    # 2. Preprocess
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    # ... apply tokenizer and target scaling ...
    
    # 3. Folds
    folds = create_folds(new_df)
    
    all_models = []
    oof_preds = []
    
    # Optional: Pseudo-labelling loop would wrap around the standard training loop here
    
    # 4. Train Loop
    for fold in range(Config.n_splits):
        fold_models = []
        for s in range(Config.num_seeds):
            model, oof = train_one_fold(fold, old_df, new_df, Config, seed=Config.seed + s)
            fold_models.append(model)
        # Average OOF across seeds
        all_models.extend(fold_models)
        
    # 5. Threshold Optimization
    # ... aggregate true labels and seed-averaged OOF floats ...
    best_thresholds = optimize_thresholds(true_labels, oof_floats)
    
    # 6. Inference
    final_preds = inference(all_models, test_df, best_thresholds)
    
    # 7. Save
    submission = pd.DataFrame({'essay_id': test_df['essay_id'], 'score': final_preds})
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1.  **Most impactful techniques:**
    * **Pre-train / Fine-tune Pipeline:** Training first on the "old" external dataset and fine-tuning exclusively on the "new" competition data yielded the largest single jump (+0.015 LB) and aligned CV with LB.
    * **Custom Threshold Optimization:** Using Scipy's Powell optimizer on `1 - QWK` to convert continuous floats to distinct integers handles the severe tail-class imbalance.
    * **Multi-Seed Averaging:** Training every model configuration with 3 different seeds and averaging the outputs (and thresholds) stabilizes predictions massively.

2.  **Secondary improvements:**
    * **Pseudo-labelling:** Passing the "old" dataset through the fine-tuned model and retraining on an average of the true/predicted labels (+0.004 to 0.007 LB).
    * **Ensembling (Simple Average):** Averaging across DeBERTa-large variations prevents the overfitting commonly seen when applying complex stackers or weighted blends to 4.5k samples.

3.  **Minor tricks:**
    * **Differential Learning Rates:** Setting distinct, lower learning rates for the model torso vs. the classification head.
    * **Custom Token Additions:** Injecting `\n` and a blank space into the tokenizer.
    * **Weight Initialization:** Utilizing normal distribution initialization for the regression head instead of PyTorch's defaults.