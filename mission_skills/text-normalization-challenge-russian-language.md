## 1. Problem Understanding
* **Task Type:** Sequence-to-Sequence (Seq2Seq) text generation. Specifically, translating raw text into normalized text (e.g., converting dates, numbers, and abbreviations into their spoken word equivalents).
* **Evaluation Metric:** Exact String Match accuracy or Character Error Rate (CER).
* **Key Challenges:** * Extreme class imbalance between "self" tokens (words that remain exactly the same) and "non-self" tokens (words requiring normalization).
    * Handling out-of-vocabulary numerical transformations.
    * Managing specific language nuances (Russian morphological rules).

## 2. Data Pipeline (Code-Oriented)
* **`load_data()`**: Parse the raw `.tsv` or text files. Iterate through the lines and apply conditional downsampling: keep $100\%$ of non-self tokens and randomly sample $10\%$ of self-tokens to balance the dataset. Combine into a training corpus of ~5M examples.
* **`preprocess()`**: 
    * Initialize two Byte-Pair Encoding (BPE) models (e.g., using `sentencepiece` or `tokenizers`).
    * **Source BPE:** Train a 20,000 token vocabulary. **Crucial logic:** Pre-filter or mask the training text to ensure absolute zero digits ($0-9$) are merged into BPE subwords (digits must remain isolated or mapped to specific character tokens).
    * **Target BPE:** Train a vocabulary keeping only tokens with a frequency $> 2$. This will result in a ~2,000 token dictionary.
* **`feature_engineering()`**: Encode the raw source and target strings into sequences of BPE integers. Append `<EOS>` and `<BOS>` tokens. Pad all sequences to a fixed maximum length (e.g., max 50 tokens) to allow for batched tensor operations.
* **`split_folds()`**: Implement an 80/20 train-validation split (or 5-fold CV if compute allows). Ensure there is no data leakage between sentences.

## 3. Model Design
* **`build_model()`**: Implement a Fully Convolutional Sequence-to-Sequence model (based on the FAIR architecture). 
    * **Model Type:** CNN-based Encoder-Decoder with Attention.
    * **Encoder:** Embedding layer followed by multiple stacked 1D Convolutions (`nn.Conv1d`) using Gated Linear Units (GLU) for activations and residual connections.
    * **Decoder:** Similar stacked 1D Convolutions, incorporating multi-step attention mechanisms that calculate attention scores at each convolutional layer.
    * **Pretrained Usage:** None. Embeddings and weights are initialized from scratch.

## 4. Training Strategy
* **`train_one_fold()`**: Standard PyTorch training loop over the DataLoader. 
* **Loss Function:** `CrossEntropyLoss` with the `ignore_index` set to the padding token ID.
* **Optimizer / Params:** Use Adam or Nesterov Accelerated Gradient (NAG). Apply learning rate scheduling (e.g., `ReduceLROnPlateau` or `StepLR`) to anneal the LR over the targeted 13 epochs.
* **Tricks:** * Use Automatic Mixed Precision (AMP) to maximize batch sizes on GPUs.
    * Apply gradient clipping (e.g., `max_norm=0.1`) to prevent exploding gradients in deep CNN architectures.
    * Use Teacher Forcing during decoding.

## 5. Validation Strategy
* **Cross-Validation Logic:** Evaluate the validation set at the end of every epoch. Track both the loss and the actual text accuracy.
* **OOF Generation:** Generate Out-Of-Fold predictions using an inference decoding method (like beam search). Compare the decoded predicted BPE string against the ground truth string.

## 6. Inference Pipeline
* **`predict()`**: Utilize beam search (e.g., beam width 5) or greedy decoding to generate the sequence of target BPE tokens.
* **TTA / Ensemble:** Test-Time Augmentation is not easily applicable here. For ensembling, average the output probabilities of multiple model checkpoints (e.g., the last 3 epochs).
* **`post_process()`**: Decode the BPE integer array back into a human-readable string. Ensure no extra spaces are added around punctuation. (Optional enhancement: apply a dictionary-based transliteration correction step here to catch known edge-case errors).

## 7. Key Tricks (ACTIONABLE)
* **BPE Digit Exclusion (CRITICAL):** * `if char.isdigit(): force_split()`
* **Target Dictionary Filtering:** * `if token_count <= 2: replace_with_UNK()`
* **Data Downsampling:** * `if label == "SELF" and random.random() > 0.1: continue`
* **Data Volume Check:** * `if len(train_data) < 5000000: load_more_data()` -> The model benefits heavily from larger data sizes; scaling from 3M to 5.5M yields significant improvements.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
class Config:
    seed = 42
    src_vocab_size = 20000
    tgt_vocab_min_freq = 3
    batch_size = 64
    epochs = 13
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# --- DATA PIPELINE ---
def load_data(file_paths):
    """Reads TSV files, keeps 100% of non-self and samples 10% of self-tokens."""
    pass

def preprocess(df, config):
    """Trains BPE tokenizers. Excludes digits from source BPE, applies freq filter to target."""
    pass

def feature_engineering(df, src_bpe, tgt_bpe):
    """Encodes text to integers, pads sequences, returns a PyTorch Dataset."""
    pass

def create_folds(df, n_splits=5):
    """Splits data into train/validation sets."""
    pass

# --- MODEL ARCHITECTURE ---
class ConvEncoder(nn.Module):
    """1D CNN Encoder with GLU activations."""
    pass

class ConvDecoder(nn.Module):
    """1D CNN Decoder with Attention."""
    pass

class ConvSeq2Seq(nn.Module):
    """Wraps Encoder and Decoder into the FAIR Seq2Seq architecture."""
    pass

def build_model(config):
    """Instantiates the ConvSeq2Seq model and moves to device."""
    pass

# --- TRAINING & VALIDATION ---
def train_one_fold(fold, train_loader, val_loader, model, criterion, optimizer, config):
    """Executes the training loop with AMP and Gradient Clipping for 'config.epochs'."""
    pass

def validate(model, val_loader, criterion, config):
    """Evaluates the model on validation data, calculates exact match metric."""
    pass

# --- INFERENCE ---
def inference(model, test_loader, tgt_bpe, config):
    """Runs beam search / greedy decoding and maps back to text."""
    pass

def post_process(predictions):
    """Applies transliteration rules to fix common errors."""
    pass

# --- MAIN EXECUTION ---
def main():
    seed_everything(Config.seed)
    
    # 1. Data loading
    raw_data = load_data(["data/ru_train.csv"])
    
    # 2. Tokenizer setup
    data, src_bpe, tgt_bpe = preprocess(raw_data, Config)
    
    # 3. Dataset creation
    dataset = feature_engineering(data, src_bpe, tgt_bpe)
    train_df, val_df = create_folds(dataset)
    
    # 4. DataLoader setup
    train_loader = DataLoader(train_df, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_df, batch_size=Config.batch_size, shuffle=False)
    
    # 5. Model, Loss, Optimizer
    model = build_model(Config)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 0 = PAD
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    
    # 6. Train
    best_model = train_one_fold(1, train_loader, val_loader, model, criterion, optimizer, Config)
    
    # 7. Inference
    # test_raw = load_data(["data/ru_test.csv"])
    # test_loader = DataLoader(...)
    # preds = inference(best_model, test_loader, tgt_bpe, Config)
    # final_text = post_process(preds)
    # save_submission(final_text)

if __name__ == "__main__":
    # main()
    pass
```

## 9. Strategy Priority

1.  **Most Impactful Techniques:** Custom vocabulary generation rules (protecting digits from BPE merging in the source, keeping $>2$ freq in the target) and maximizing dataset size to 5M+ tokens.
2.  **Secondary Improvements:** Class balancing via downsampling the "self" tokens to $10\%$. Utilizing the FAIR Fully Convolutional Seq2Seq architecture instead of a standard RNN to capture local context efficiently.
3.  **Minor Tricks:** Implementation of a post-processing transliteration dictionary to clean up character-level generation errors.