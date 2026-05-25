Here is the executable solution blueprint derived from the 1st place solution. This document is engineered specifically to guide a code-generation LLM in writing a complete, single-file PyTorch script.

## 1. Problem Understanding
* **Task Type:** Extractive Question Answering / Span Prediction (Extracting text substrings based on sentiment).
* **Evaluation Metric:** Word-level / Character-level Jaccard Similarity.
* **Key Challenges:** * Standard Cross-Entropy does not directly optimize Jaccard.
    * Token-level predictions must be mapped back to character-level strings accurately.
    * Handling noisy, short text where standard Transformer tokenization creates boundary alignment issues.

## 2. Data Pipeline (Code-Oriented)
* **`load_data()`**: Read `train.csv` and `test.csv`. Fill NA values with empty strings.
* **`preprocess(df)`**: 
    * Append sentiment token to the text (e.g., `text + " " + sentiment`). For specific architectures, prepend auxiliary sentiment.
    * Use a custom `merges.txt` (for RoBERTa) to handle edge-case tokenizations.
    * Create robust token-to-character offset mappings (critical for 2nd level models).
* **`feature_engineering(df)`**: 
    * Convert target substrings into `start_idx` and `end_idx` token labels.
    * Generate Jaccard-based Soft Labels: Calculate the Jaccard overlap of every token with the ground truth span, smooth with a square term, and prepare for KL Divergence.
    * Apply Sequence Bucketing (grouping texts of similar lengths) to minimize padding.
* **`split_folds(df)`**: 5-Fold Stratified K-Fold based on the `sentiment` column to ensure class balance across folds.

## 3. Model Design
* **`build_model(config)`**: Create a flexible factory function for Level 1 Transformers.
    * **Base Architectures:** `roberta-base-squad2`, `bert-base-uncased`, `distilbert-base-uncased`. (Load pre-trained SQuAD weights if available).
    * **Head Architecture:** Extract the last $n$ (e.g., 3) hidden states. Concatenate or average them.
    * **Multi-Sample Dropout (MSD):** Apply 5 different dropout masks to the concatenated hidden states, pass each through the final Linear/CNN layers, and average the outputs.
* **`build_level2_model(config)`**: 
    * **Architecture:** 1D CNN, Char-RNN, or WaveNet.
    * **Inputs:** Character-level probabilities (start and end) predicted by the Level 1 models.

## 4. Training Strategy
* **`train_one_fold(fold, train_loader, val_loader)`**: Standard PyTorch training loop using Mixed Precision (`torch.cuda.amp`).
* **Loss Functions:** * *Primary:* Custom Jaccard KL-Divergence Loss (calculates KL divergence between model logits and custom Jaccard soft-labels).
    * *Secondary:* Smoothed Categorical Cross-Entropy.
    * Combine them using a weighting parameter $\alpha$ (e.g., $\alpha = 0.3$).
* **Optimizer / Params:** * AdamW optimizer.
    * Layer-wise Learning Rate Decay (LLRD): Lower LR for bottom layers, higher LR for the classification head (e.g., 1e-4 for head, 3e-5 to 7e-5 for backbone).
    * Linear decay schedule (no warmup).
    * Weight decay = 0.001. Batch size = 32 (or highest power of 2 that fits). Epochs = 3 to 5.
* **Tricks:** Stochastic Weight Averaging (SWA) applied at the end of training for generalization.

## 5. Validation Strategy
* **Cross-Validation Logic:** Standard 5-fold evaluation. 
* **OOF Generation:** During validation, convert token-level logit predictions into character-level probabilities using the offset mappings. Save these character-level arrays as `oof_preds` to train the Level 2 Char-NN models.

## 6. Inference Pipeline
* **`predict(models, dataloader)`**: 
    * Run test data through all Level 1 models.
    * Average the character-level probabilities (Ensemble).
    * Feed ensembled probabilities into the trained Level 2 Char-NN models.
* **`post_process(start_probas, end_probas, text, sentiment)`**: 
    * Traverse the top 20 `start_idx` and `end_idx` pairs. Pick the pair with the highest combined probability where `start_idx < end_idx`.
    * **Overrides:** If the sentiment is "neutral", or if no valid `start < end` pair is found, force the output to be the entire original text.

## 7. Key Tricks (ACTIONABLE)
* **If** training a Transformer for span extraction -> **Do** apply Multi-Sample Dropout to the last 3 concatenated hidden layers to prevent overfitting.
* **If** standard Cross-Entropy isn't maximizing Jaccard -> **Do** convert ground truth spans into token-level Jaccard scores, square them, and use KL Divergence loss.
* **If** ensemble combinations are highly correlated -> **Do** output character-level probabilities from Level 1 and use them as inputs to a Level 2 Char-CNN/RNN (Stacking).
* **If** test set is large enough -> **Do** generate pseudo-labels. Keep samples where `(max(start_prob) + max(end_prob)) / 2 > 0.35` and add them to the train set.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.model_selection import StratifiedKFold

# --- CONFIGURATION ---
class Config:
    seed = 42
    model_name = "roberta-base"
    max_len = 120
    epochs = 5
    batch_size = 32
    learning_rate = 3e-5
    head_lr = 1e-4
    num_folds = 5

def seed_everything(seed):
    # Set seeds for OS, NumPy, PyTorch (CPU/CUDA) to ensure reproducibility.
    pass

# --- DATA MANIPULATION ---
def load_data():
    # Load CSVs, handle NAs, return train_df and test_df.
    pass

def preprocess(df):
    # Append sentiment token, clean basic text, initialize tokenizer.
    pass

def feature_engineering(df, tokenizer):
    # Map text to tokens, generate offset mappings, create Jaccard soft-labels.
    pass

def create_folds(df):
    # Use StratifiedKFold on the 'sentiment' column, assign fold numbers.
    pass

class TweetDataset(Dataset):
    # PyTorch Dataset returning input_ids, attention_mask, offsets, and soft labels.
    pass

# --- LOSS & MODULES ---
class JaccardKLLoss(nn.Module):
    # Implement custom loss: KL Divergence between logits and Jaccard-based soft labels.
    pass

class MultiSampleDropout(nn.Module):
    # Apply multiple dropout masks to the input and average the outputs.
    pass

# --- MODELS ---
class Level1Transformer(nn.Module):
    # Load backbone, extract last N hidden states, apply MSD, output start/end logits.
    pass

class Level2CharNN(nn.Module):
    # 1D CNN or WaveNet that takes character-level probas and outputs final spans.
    pass

# --- TRAINING LOOP ---
def get_optimizer_grouped_parameters(model, config):
    # Implement Layer-wise Learning Rate Decay (LLRD).
    pass

def train_one_fold(fold, train_loader, val_loader, config):
    # Initialize model, optimizer, AMP scaler.
    # Run training loop, calculate JaccardKL loss.
    # Return trained model and OOF character-level probabilities.
    pass

def train_level2_model(oof_probas, train_targets, config):
    # Train the Char-NN using the OOF predictions from Level 1.
    pass

# --- INFERENCE ---
def post_process(start_preds, end_preds, text, sentiment):
    # Traverse top 20 indices where start < end. Apply neutral sentiment override.
    pass

def predict(l1_models, l2_model, test_loader):
    # Run test data through L1 -> average probas -> run through L2 -> post_process.
    pass

# --- MAIN ---
def main():
    seed_everything(Config.seed)
    train_df, test_df = load_data()
    
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    train_df = preprocess(train_df)
    train_df = feature_engineering(train_df, tokenizer)
    train_df = create_folds(train_df)
    
    l1_models = []
    oof_predictions = np.zeros(...) # Initialize array for character probas
    
    # Stage 1: Train Level 1 Transformers
    for fold in range(Config.num_folds):
        train_loader, val_loader = ... # build dataloaders
        model, oof_fold = train_one_fold(fold, train_loader, val_loader, Config)
        l1_models.append(model)
        oof_predictions[val_indices] = oof_fold
        
    # Stage 2: Train Level 2 Char-NN Stacker
    l2_model = train_level2_model(oof_predictions, train_df['targets'], Config)
    
    # Stage 3: Inference
    test_loader = ... # build test dataloader
    final_preds = predict(l1_models, l2_model, test_loader)
    
    # Save submission
    submission = pd.DataFrame({'textID': test_df['textID'], 'selected_text': final_preds})
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)
1.  **Most Impactful Techniques:** 2nd Level Char-NN Stacker (utilizing character-level probabilities instead of token logic), Custom Jaccard-based KL Divergence Loss.
2.  **Secondary Improvements:** Multi-Sample Dropout, Layer-wise Learning Rate Decay (Discriminative Learning), `[sentiment]` token concatenation.
3.  **Minor Tricks:** Pseudo-labeling (only 0.001 boost, high risk of leakage if done wrong), SWA, custom `merges.txt`.