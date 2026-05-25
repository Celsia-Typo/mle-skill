Here is the structured, code-oriented blueprint designed to guide an LLM in generating a single-file Python script for this 1st place solution.

## 1. Problem Understanding
* **Task type:** NLP Sentence/Phrase Pair Scoring (Regression framed as Semantic Similarity).
* **Evaluation metric:** Pearson Correlation Coefficient.
* **Key challenges:** Handling domain-specific patent vocabulary, modeling slight semantic nuances between multiple targets associated with the same anchor, and preventing overfitting on short text pairs.

## 2. Data Pipeline (Code-Oriented)
* **`load_data()`**: Load train and test CSVs. Map the `context` code (e.g., "A47") to its corresponding full text description (`CPC_TEXT`) using an external or generated dictionary.
* **`preprocess(df)`**: Create a `sector` column by extracting the first character of the `context` column (e.g., `context[0]` -> "F21" becomes "F").
* **`feature_engineering(df)`**:
    * Create a mapping dictionary grouping all targets by `['anchor', 'context']`.
    * Create a secondary mapping dictionary grouping all targets by `['anchor', 'sector']`.
    * Format the text input: Construct a string following the pattern `anchor [SEP] target [SEP] cpc_text [SEP] other_targets`.
    * **Crucial Rule**: When appending `other_targets`, exclude the current row's `target` string.
* **`split_folds(df)`**: Implement GroupKFold grouped by the `anchor` column, but stratify the folds by the `score` distribution. Ensure that any words appearing in *both* the anchor and the target are isolated to the same fold to prevent data leakage.

## 3. Model Design
* **`build_model(config)`**:
    * **Backbone**: Initialize a HuggingFace Transformer (e.g., `microsoft/deberta-v3-large`, `anferico/bert-for-patents`).
    * **Embedding Layer**: Freeze the BERT embedding layer (do not update weights during training).
    * **Header**: Feed the transformer output into a Bidirectional LSTM (Bi-LSTM).
    * **Scaling Rule**: For weaker models (like `bert-for-patents`), multiply the RNN output dimension by 2 (e.g., expand 1024 to 2048).
    * **Pooling**: Apply Linear Attention Pooling on top of the Bi-LSTM outputs.
    * **Output**: Pass the pooled output through a final Fully Connected (Linear) layer to predict a single scalar score.

## 4. Training Strategy
* **`train_one_fold(fold, train_df, val_df, config)`**:
    * **Epochs**: Train for exactly 5 epochs.
    * **Loss Function**: Pearson Correlation Loss (differentiable implementation).
    * **Optimizer**: AdamW. Apply differential learning rates:
        * Transformer backbone: `2e-5` (or `3e-5` depending on model).
        * Bi-LSTM and Custom Head: `1e-3`.
    * **Scheduler**: Linear or Cosine schedule (depends on the backbone).
    * **Dynamic Trick**: Inside the PyTorch `Dataset` `__getitem__` method, randomly shuffle the list of `other_targets` every time a sample is fetched during training.
    * **Adversarial Training**: Initialize AWP (Adversarial Weight Perturbation) and enable it strictly starting from the 2nd epoch.

## 5. Validation Strategy
* **`validate(model, val_loader)`**:
    * Run standard inference loop without gradient calculation.
    * Calculate the Pearson correlation score between predictions and ground truths.
    * Save Out-Of-Fold (OOF) predictions for the evaluation dataset.
    * Save the model state dictionary with the highest validation score.

## 6. Inference Pipeline
* **`predict(models, test_loader)`**: Generate predictions for a given test set using a list of loaded fold models.
* **`post_process(predictions)`**: Apply MinMax scaling to the raw prediction array of *each individual model* to normalize outputs to a 0-1 range before ensembling.
* **`ensemble(all_model_preds)`**:
    * Do NOT use a simple mean average. Use a weighted average based on model strength.
    * Implement logic combining 5 Fold models + 1 Full-Train model per architecture.
    * Multiply the Full-Train model's output weight by 2 relative to the fold models.
    * Apply backbone weights (e.g., DeBERTa = 1.0, BERT-for-patents = 0.4).

## 7. Key Tricks (ACTIONABLE)
* **If building the dataset** → concatenate grouped targets separated by `[SEP]`, exclude the current target, and dynamically shuffle them in the `__getitem__` method.
* **If configuring the optimizer** → strictly isolate parameter groups: embed layer (requires `requires_grad=False`), transformer layers (lr=`2e-5`), and LSTM/Head (lr=`1e-3`).
* **If training loop reaches Epoch 2** → toggle `awp.step()` inside the backward pass.
* **If ensembling final predictions** → run `MinMaxScaler().fit_transform(preds)` on every model array prior to applying weighted sums.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
# ... other standard imports ...

def seed_everything(seed):
    """Sets random seeds for reproducibility across numpy, torch, and python."""
    pass

def load_data():
    """Reads train.csv, test.csv, and maps context codes to CPC textual descriptions."""
    pass

def preprocess(df):
    """Extracts 'sector' from the 'context' column (e.g., index 0)."""
    pass

def feature_engineering(df, is_train=True):
    """Groups targets by context/sector, concatenates [SEP] strings, and generates final text column."""
    pass

def create_folds(df):
    """Executes GroupKFold grouped by anchor, stratifying on score, avoiding overlapping word leakage."""
    pass

class PatentDataset(torch.utils.data.Dataset):
    """Handles tokenization and dynamically shuffles grouped targets during training."""
    pass

class CustomModel(nn.Module):
    """
    Constructs the model:
    - Freezes transformer embeddings.
    - Applies Bi-LSTM on transformer output.
    - Applies Linear Attention Pooling.
    - Final Linear layer for score prediction.
    """
    pass

class AWP:
    """Implements Adversarial Weight Perturbation logic."""
    pass

def pearson_loss(predictions, targets):
    """Computes differentiable Pearson correlation loss."""
    pass

def train_one_fold(fold, train_df, val_df, config):
    """
    Standard training loop.
    Implements differential learning rates.
    Activates AWP logic starting from epoch 2.
    Returns the best model for the fold.
    """
    pass

def validate(model, val_loader):
    """Runs inference on validation fold, computes Pearson score, returns OOF predictions."""
    pass

def predict(models, test_loader):
    """Generates raw predictions on the test set from a list of trained models."""
    pass

def post_process_and_ensemble(pred_dict):
    """Applies MinMax scaling to each model's predictions and calculates the weighted average."""
    pass

def main():
    # 1. Setup & Load
    seed_everything(42)
    train_df, test_df = load_data()
    
    # 2. Preprocess & Feature Engineering
    train_df = preprocess(train_df)
    train_df = feature_engineering(train_df)
    train_df = create_folds(train_df)
    
    test_df = preprocess(test_df)
    test_df = feature_engineering(test_df, is_train=False)

    # 3. Train Fold Models
    fold_models = []
    for fold in range(5):
        model = train_one_fold(fold, train_df, val_df, config)
        fold_models.append(model)
    
    # 4. Train Full Data Model
    full_train_model = train_one_fold(fold=-1, train_df=train_df, val_df=None, config)
    
    # 5. Inference
    fold_preds = predict(fold_models, test_loader)
    full_preds = predict([full_train_model], test_loader)
    
    # 6. Ensemble and Submit
    # (Repeat steps 3-5 for different backbones, store in pred_dict)
    final_preds = post_process_and_ensemble(pred_dict)
    
    # Save submission
    # ...

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1.  **Most impactful techniques:** Target grouping by anchor/context injected into the input string; Bi-LSTM with Linear Attention pooling added to the model head.
2.  **Secondary improvements:** MinMax scaling prior to ensembling; 5-fold + Full Train strategy (with Full Train weight = 2); Adversarial Weight Perturbation (AWP).
3.  **Minor tricks:** Target grouping by sector (for ensemble diversity); Freezing BERT embeddings; Differential learning rates; Expanding RNN dimensions for weaker models.