# Blueprint: Jigsaw Unintended Bias in Toxicity Classification

> ✅ **Nature of this notebook: Full end-to-end pipeline.**
> Covers tokenization, dual pretrained embedding construction, identity-aware sample weighting, Bidirectional CuDNNLSTM with multi-output training, checkpoint ensemble with exponential weights, and `submission.csv` generation.
> Source file: `jigsaw-unintended-bias-in-toxicity-classification.py`

---

## Competition Details

| Field | Value |
|---|---|
| Competition | [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification) |
| Task | Binary toxicity prediction for online comments, penalizing bias against identity groups |
| Input | Raw comment text |
| Target | `target` — continuous [0, 1] toxicity score (thresholded at 0.5) |
| Evaluation | Custom bias-aware AUC: weighted combination of overall AUC and per-identity-group subgroup AUCs |

---

## Dependencies

| Library | Role |
|---|---|
| `keras` | Model building and training (CuDNNLSTM requires GPU) |
| `gensim.models.KeyedVectors` | Loading pretrained word vectors in gensim format |
| `keras.preprocessing.text.Tokenizer` | Text tokenization and vocabulary building |
| `keras.preprocessing.sequence.pad_sequences` | Padding token sequences to fixed length |
| `numpy`, `pandas` | Data manipulation |

**GPU required:** `CuDNNLSTM` is a CUDA-optimized LSTM variant — the model will not run on CPU.

---

## Data

| File | Rows | Key Columns |
|---|---|---|
| `train.csv` | ~1.8M | `comment_text`, `target`, `severe_toxicity`, `obscene`, `identity_attack`, `insult`, `threat`, + 9 identity columns |
| `test.csv` | ~97K | `id`, `comment_text` |

**Identity columns (9):** `male`, `female`, `homosexual_gay_or_lesbian`, `christian`, `jewish`, `muslim`, `black`, `white`, `psychiatric_or_mental_illness`

**Auxiliary target columns (6):** `target`, `severe_toxicity`, `obscene`, `identity_attack`, `insult`, `threat`

**Binarization of identity + target columns:**
```python
for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:
    train_df[column] = np.where(train_df[column] >= 0.5, True, False)
```
- Identity and target columns are originally continuous [0, 1]; thresholded at 0.5 for sample weighting

---

## Pipeline

### Step 1 — Tokenization

```python
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n""'\'∞θ÷α•à−β∅³π'₹´°£€\×™√²—'
MAX_LEN = 220

tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE, lower=False)
tokenizer.fit_on_texts(list(x_train) + list(x_test))  # vocabulary built on train + test

x_train = sequence.pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=MAX_LEN)
x_test  = sequence.pad_sequences(tokenizer.texts_to_sequences(x_test),  maxlen=MAX_LEN)
```

- `lower=False`: case is preserved during tokenization (embedding lookup tries original case first, then lowercase)
- Vocabulary fitted on **both train and test** to maximize embedding coverage
- Custom filter string removes punctuation, special Unicode characters, and mathematical symbols
- Sequences truncated/padded to `MAX_LEN=220`

### Step 2 — Embedding Matrix Construction

```python
EMBEDDING_FILES = [
    '../input/gensim-embeddings-dataset/crawl-300d-2M.gensim',   # FastText Common Crawl
    '../input/gensim-embeddings-dataset/glove.840B.300d.gensim'  # GloVe 840B
]

def build_matrix(word_index, path):
    embedding_index = KeyedVectors.load(path, mmap='r')   # memory-mapped for efficiency
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        for candidate in [word, word.lower()]:            # try original case, then lowercase
            if candidate in embedding_index:
                embedding_matrix[i] = embedding_index[candidate]
                break
    return embedding_matrix

embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1
)
# Shape: (vocab_size + 1, 600) — 300 FastText + 300 GloVe concatenated
```

- `mmap='r'`: memory-maps the gensim file to avoid loading the full 5GB+ embedding into RAM
- Case fallback: tries the original token first, then lowercased version — improves coverage for capitalized words
- Words not found in either embedding remain as zero vectors

### Step 3 — Identity-Aware Sample Weighting

The core bias-mitigation technique — upweights comments involving identity groups to reduce model bias:

```python
sample_weights = np.ones(len(x_train), dtype=np.float32)
sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)            # +1 per identity mentioned
sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)   # toxic + no identity
sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5  # non-toxic + identity (5×)
sample_weights /= sample_weights.mean()   # normalize to mean = 1
```

**Weight components explained:**

| Condition | Weight Boost | Rationale |
|---|---|---|
| Comment mentions any identity | +1 per identity | Ensures model sees identity-mentioning comments |
| Toxic + no identity mentioned | + (9 − n_identities) | Reinforces true toxic signal |
| **Non-toxic + identity mentioned** | **+ 5 × n_identities** | Strongest boost: prevents false positives on identity-mentioning benign comments |

The 5× multiplier on non-toxic identity comments directly addresses the competition's bias metric — models that flag comments mentioning minorities as toxic are penalized heavily.

### Step 4 — Model Architecture

```python
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS   # 512

def build_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),       # 256-dim: max over time
        GlobalAveragePooling1D()(x),   # 256-dim: mean over time
    ])                                 # → 512-dim combined representation

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])  # residual block 1
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])  # residual block 2

    result     = Dense(1, activation='sigmoid')(hidden)                # main: toxicity
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)  # aux: 6 targets

    model = Model(inputs=words, outputs=[result, aux_result])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model
```

**Architecture summary:**

| Layer | Output Shape | Notes |
|---|---|---|
| Embedding | `(batch, 220, 600)` | Frozen; concatenated FastText + GloVe |
| SpatialDropout1D(0.2) | `(batch, 220, 600)` | Drops entire embedding dimensions, not individual tokens |
| BiLSTM × 2 (128 units each) | `(batch, 220, 256)` | CuDNNLSTM: GPU-only CUDA implementation |
| Concat(MaxPool, AvgPool) | `(batch, 512)` | Captures both peak and average signal across sequence |
| Residual Dense × 2 (512, relu) | `(batch, 512)` | Skip connections prevent gradient vanishing |
| Output (sigmoid) | `(batch, 1)` | Main toxicity prediction |
| Aux output (sigmoid) | `(batch, 6)` | `target`, `severe_toxicity`, `obscene`, `identity_attack`, `insult`, `threat` |

- `trainable=False` on Embedding: pretrained vectors are frozen
- Multi-output model: both heads trained jointly with equal loss weight
- `sample_weight` applied only to main output; auxiliary output uses `np.ones_like(sample_weights)`

### Step 5 — Training with Checkpoint Ensemble

```python
NUM_MODELS = 2
EPOCHS = 4
checkpoint_predictions = []
weights = []

for model_idx in range(NUM_MODELS):
    model = build_model(embedding_matrix, y_aux_train.shape[-1])
    for global_epoch in range(EPOCHS):
        model.fit(
            x_train,
            [y_train, y_aux_train],
            batch_size=512,
            epochs=1,           # one epoch at a time to save checkpoints
            sample_weight=[sample_weights.values, np.ones_like(sample_weights)]
        )
        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
        weights.append(2 ** global_epoch)   # exponential weights: [1, 2, 4, 8]
```

- **2 models** trained from different random initializations → 8 total checkpoint predictions (2 × 4)
- **Epoch-by-epoch training** (`epochs=1` each call) allows saving a test prediction after every epoch
- **Exponential checkpoint weights** `[1, 2, 4, 8]`: later epochs receive exponentially more weight, reflecting that later checkpoints are better trained
- No explicit validation set; no early stopping

### Step 6 — Weighted Ensemble and Submission

```python
predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
# weights = [1, 2, 4, 8, 1, 2, 4, 8] for model_0 epochs + model_1 epochs

submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)
```

- Final predictions: weighted average of 8 checkpoint arrays
- Output is a continuous probability in [0, 1] (not thresholded) — competition expects raw scores for AUC evaluation

---

## Output

| File | Description |
|---|---|
| `submission.csv` | `id`, `prediction` — continuous toxicity probability per test comment |

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| Dual embeddings (FastText + GloVe concatenated) | Different embeddings capture complementary semantic properties; FastText handles OOV via subword; GloVe captures global co-occurrence statistics; concatenation consistently outperforms single embeddings |
| `lower=False` + case fallback in `build_matrix` | Toxic language often uses unusual capitalization ("HATE"); preserving case improves toxicity signal; fallback ensures OOV coverage |
| Vocabulary fitted on train + test | Maximizes embedding hit rate for test vocabulary; avoids OOV tokens in test set that were unseen during tokenizer fitting |
| Identity-aware sample weighting (5× non-toxic) | Directly counters the most common bias failure mode: predicting comments about minorities as toxic just because they mention identity words |
| Auxiliary multi-task outputs | Joint training on related toxicity subtypes (obscene, insult, threat) acts as regularization and improves main target generalization |
| Residual Dense connections | Prevents gradient vanishing in deep MLP head; allows model to learn both identity and incremental transformations |
| Concat MaxPool + AvgPool | MaxPool captures the single most toxic signal; AvgPool captures the overall tone — combining both gives a richer sentence representation than either alone |
| Exponential checkpoint weighting (`2^epoch`) | Later checkpoints are better trained; exponential weighting is a simple heuristic that emphasizes the final state while still benefiting from diversity across epochs |
| `CuDNNLSTM` | CUDA-optimized LSTM variant — 3–5× faster than standard LSTM on GPU; requires NVIDIA GPU |

---

## SOTA Gap

| Aspect | This Notebook | Competition SOTA |
|---|---|---|
| Backbone | Bi-LSTM (2 layers, 128 units) | BERT-Large, RoBERTa, XLNet fine-tuned |
| Embedding | Frozen FastText + GloVe (600-dim) | Contextual embeddings (transformer subword tokens) |
| Bias mitigation | Sample weighting only | Sample weighting + identity-stratified loss + post-processing calibration |
| Validation | None (no val split) | Stratified holdout with per-identity AUC monitoring |
| Models ensembled | 2 models × 4 checkpoints | 5–10 transformer models (BERT + RoBERTa + XLNet + GPT-2) |
| Public LB AUC | ~0.930 (referenced kernel score) | ~0.947 (top solutions) |
| Training data use | Full 1.8M rows | Full 1.8M rows + external Jigsaw toxicity datasets |

---

## Suggested Improvements

1. **Replace Bi-LSTM with fine-tuned BERT/RoBERTa** — transformer contextual embeddings capture long-range dependencies and subword toxicity signals; BERT-base alone closes ~0.010 AUC gap over this baseline
2. **Add a validation split with per-identity AUC tracking** — training without any validation makes it impossible to monitor bias metrics during training; a stratified holdout (by identity presence and toxicity) is essential for tuning
3. **Tune the 5× non-toxic identity weight** — the multiplier is a fixed heuristic; grid searching over values (3×, 5×, 7×, 10×) on a validation bias metric would find a better operating point
4. **Add stratified KFold** — with 1.8M training rows and 2 models, KFold is feasible; fold-level OOF predictions provide a better estimate of the bias metric and enable more reliable ensembling
5. **Apply loss weighting instead of sample weighting** — using `class_weight` per identity subgroup as a loss multiplier (rather than inflating sample counts) is numerically cleaner and avoids gradient scale distortion
6. **Add external toxicity data** — the original Jigsaw Toxic Comment Classification dataset (~160K rows) and Civil Comments dataset can be used as additional training signal; top solutions used both
7. **Use `SWA` (Stochastic Weight Averaging)** — instead of exponential checkpoint weighting, SWA averages model weights rather than predictions, which often finds flatter minima and generalizes better
8. **Upgrade from `CuDNNLSTM` to `tf.keras.layers.LSTM(units, implementation=2)`** — `CuDNNLSTM` is deprecated in TF 2.x; `implementation=2` on `tf.keras.layers.LSTM` uses the same fused CUDA kernel
