Here is the structured, executable solution blueprint derived from the 1st place solution write-up. 

## 1. Problem Understanding
- **Task type**: Document-level Question Answering (Classification + Span Extraction).
- **Evaluation metric**: Micro F1-score for both Long Answer and Short Answer predictions (evaluated via thresholding).
- **Key challenges**: 
  - Massive class imbalance (40 million total candidates, most are easy negatives).
  - The disconnect between candidate-level classification (easy) and document-level classification (difficult).
  - Ensembling models that utilize entirely different tokenizers (BERT vs. ALBERT).

## 2. Data Pipeline (Code-Oriented)
- **`load_data()`**: Parse the nested JSONL format. Extract provided candidates rather than sampling text randomly from the raw document strings.
- **`preprocess()`**: 
  - Initialize the Hugging Face tokenizer. 
  - Add 9 specific HTML tags (derived from dataset statistics) as special tokens. 
  - For any HTML tag not in the top 9, map it to a newly created generic `<UNK_HTML>` token.
- **`feature_engineering()`**: 
  - Convert input text into `input_ids`, `attention_mask`, and `token_type_ids`.
  - Build a mapping array that links every subword token back to its original whitespace-separated word index.
- **`hard_negative_sampling()`**: 
  - **Pass 1**: Train a baseline model using uniform random sampling (1 positive, 1 random negative per document).
  - **Pass 2**: Run inference on the entire training set to get the `prob(answer)` for every negative candidate. 
  - Normalize these probabilities within each document to create a multinomial distribution.
  - Sample exactly one negative candidate per document per epoch based on this distribution.

## 3. Model Design
- **`build_model()`**: 
  - **Backbone**: PyTorch-based Hugging Face transformers.
  - **Classification Head**: A Linear layer mapping the `[CLS]` token output to 5 classes: `no_answer`, `long_answer_only`, `short_answer`, `yes`, `no`.
  - **Span Head**: A Linear layer mapping all token sequence outputs to 2 dimensions (start_logit, end_logit).
- **Pretrained Usage**: 
  - Model pools: `bert-base-uncased`, `bert-large-uncased-whole-word-masking` (SQuAD pre-tuned), and `albert-xxl-v2` (SQuAD pre-tuned).

## 4. Training Strategy
- **`train_one_fold()`**: Loop through the hard-negative sampled candidates.
- **Loss function**: 
  - Classification: CrossEntropyLoss / BCE for the 5-class head.
  - Span: CrossEntropyLoss for start and end logits. 
  - **Crucial Masking**: If a candidate does not contain a short answer, multiply the span loss by 0 (ignore it for backpropagation).
- **Optimizer / Params**: AdamW optimizer with a linear warmup scheduler.
- **Tricks**: Use PyTorch AMP (Automatic Mixed Precision) for `bert-large` and `albert-xxl` to fit sequences into GPU memory.

## 5. Validation Strategy
- **Cross-validation logic**: Standard K-Fold grouped by `document_id` to prevent data leakage.
- **OOF generation**: Store logits for the 5 classes and start/end span probabilities.
- **Threshold Tuning**: Run a grid search on the out-of-fold predictions to find the optimal confidence thresholds for both long and short answers to maximize the competition F1 metric.

## 6. Inference Pipeline
- **`predict()`**: Run test candidates through the network.
- **`align_tokenizers()`**: Map token-level probabilities (start/end) back to word-level (whitespace) probabilities using the map created in preprocessing. This enables direct averaging of logits across BERT and ALBERT.
- **`post_process()`**:
  - **Long Answer Score**: Calculate `1.0 - prob(no_answer)`. Assign this score to the candidate. Select the candidate with the highest score per document.
  - **Short Answer Extraction**: Force the short answer span to reside entirely within the selected top long answer candidate.
  - **Short Answer Score**: Calculate `prob(short_answer) + prob(yes) + prob(no)`.
  - **Short Answer Class**: Determine the exact sub-type by taking `argmax(prob(short_answer), prob(yes), prob(no))`.

## 7. Key Tricks (ACTIONABLE)
- **If Ensembling heterogeneous architectures** → Map subword token outputs back to uniform whitespace word indices before averaging logits.
- **If extreme negative imbalance** → Implement 2-stage hard negative sampling using baseline probabilities.
- **If computing loss on missing spans** → Explicitly zero out the start/end loss gradients to prevent the model from getting confused.
- **If thresholds dictate F1** → Disconnect final class assignments from `argmax` and apply custom tuned thresholds for `1.0 - prob(no_answer)` to trigger long answer presence.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

def seed_everything(seed=42):
    # Set seeds for reproducibility

def load_data(file_path):
    # Parse JSONL, extract document candidates and labels

def preprocess(df, tokenizer):
    # Add HTML tokens to tokenizer, generate token_ids, masks

def align_word_pieces(tokenized_data, raw_text):
    # Map subword tokens back to whitespace-separated word indices

def generate_hard_negatives(train_df, baseline_model):
    # Infer on train_df, create probability distributions, sample negatives

def create_folds(df, num_folds=5):
    # GroupKFold split based on document_id

class QADataset(Dataset):
    # PyTorch dataset yielding input_ids, masks, labels, and span targets

class TFQAModel(nn.Module):
    # Init transformer backbone
    # Define 5-class linear head
    # Define 2-class span head (start, end)
    # Forward pass returning logits

def calculate_loss(cls_logits, span_logits, cls_targets, span_targets):
    # Compute classification loss
    # Compute span loss (masked if short answer doesn't exist)
    # Return weighted combination

def train_one_fold(fold, train_df, val_df, config):
    # Initialize DataLoader with hard negative samples
    # Setup model, AdamW, scaler for AMP
    # Loop epochs, compute calculate_loss, backpropagate
    # Save best checkpoint

def validate(model, val_loader):
    # Compute predictions on validation set
    # Return document-level probabilities

def optimize_thresholds(oof_preds, labels):
    # Grid search to maximize Long F1 and Short F1

def document_level_inference(models, test_loader):
    # Predict logits
    # Call align_word_pieces to standardize token outputs
    # Ensemble by averaging word-level probabilities
    # Apply post-processing (score = 1.0 - no_answer_prob)
    # Restrict short span within top long candidate

def main():
    config = {"model_name": "bert-large-uncased-whole-word-masking", "epochs": 3}
    seed_everything()
    
    train_data, test_data = load_data("train.jsonl"), load_data("test.jsonl")
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    # Add HTML special tokens here
    
    train_df = preprocess(train_data, tokenizer)
    test_df = preprocess(test_data, tokenizer)
    
    # Pass 1: Optional baseline load for hard negatives
    train_df = generate_hard_negatives(train_df, None) 
    
    folds = create_folds(train_df)
    models = []
    
    for fold in range(5):
        model = train_one_fold(fold, folds[fold]["train"], folds[fold]["val"], config)
        models.append(model)
        
    preds = document_level_inference(models, test_df)
    # Save submission.csv based on optimal thresholds
    
if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1. **Most impactful techniques**: 
   - **Hard Negative Sampling**: Moving from uniform sampling to probability-distribution sampling fundamentally forces the model to learn document-level distinctions, bridging the gap between candidate and document evaluation.
   - **Word-Level Ensemble Mapping**: Resolving the tokenizer mismatch (BERT vs. ALBERT) at the whitespace word level to allow seamless ensembling of diverse architectures.
2. **Secondary improvements**: 
   - Initializing with SQuAD pre-tuned weights instead of base language models.
   - Constraining short-answer span extraction strictly within the boundaries of the top-scored long-answer candidate.
3. **Minor tricks**: 
   - Adding domain-specific HTML tags as vocabulary tokens.
   - Extensive threshold tweaking on `1.0 - prob(no_answer)` for Long Answer confidence.