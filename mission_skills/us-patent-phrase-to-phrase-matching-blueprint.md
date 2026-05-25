# Blueprint: us-patent-phrase-to-phrase-matching.ipynb

## Overview

This notebook is a **transformer fine-tuning tutorial and baseline** for the [US Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching) Kaggle competition. It is authored by Jeremy Howard (fast.ai) and doubles as a comprehensive NLP course notebook, teaching tokenization, validation set construction, overfitting/underfitting, and Pearson correlation — alongside building and submitting a working DeBERTa-v3-small model.

---

## Purpose

Fine-tune a pretrained transformer (`microsoft/deberta-v3-small`) on patent phrase similarity pairs to predict a continuous semantic similarity score, submitted as a regression task evaluated by Pearson correlation.

---

## Dependencies

| Library | Role |
|---|---|
| `pandas`, `numpy` | Data handling and math |
| `datasets` (HuggingFace) | Dataset and DatasetDict objects |
| `transformers` | `AutoTokenizer`, `AutoModelForSequenceClassification`, `TrainingArguments`, `Trainer` |
| `matplotlib` | Plotting (overfitting/underfitting demo) |
| `sklearn` | `PolynomialFeatures`, `LinearRegression`, California Housing dataset (for correlation demo) |
| `pathlib` | Path handling |
| `fastkaggle` | (optional) Competition data download helper |

---

## Competition Details

| Field | Value |
|---|---|
| Task | Regression — semantic similarity scoring |
| Input | Patent anchor phrase + target phrase + CPC section context code |
| Labels | Continuous score in `[0, 1]` (human-rated semantic similarity) |
| Evaluation | Pearson correlation coefficient |
| Data files | `train.csv`, `test.csv` |

---

## Data

`train.csv` columns: `id`, `anchor`, `target`, `context`, `score`

- 36,473 rows, 733 unique anchors, 106 CPC context codes, ~30,000 unique targets.
- Scores reflect human judgment on a 5-point scale collapsed to `[0, 1]`.

### Input Construction
```python
df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor
```
This single concatenated string is passed to the tokenizer, encoding all three relevant fields.

---

## Pipeline

### 1. Data Loading & EDA
```python
df = pd.read_csv(path/'train.csv')
df.describe(include='object')
```
Counts unique anchors, contexts, and targets. Provides a feel for the multi-to-multi mapping structure.

### 2. Tokenization
Model: `microsoft/deberta-v3-small`

```python
tokz = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
def tok_func(x): return tokz(x["input"])
tok_ds = ds.map(tok_func, batched=True)
```
- Tokenizer splits text into subword pieces (sentencepiece, `▁` marks word starts).
- Produces `input_ids` and `attention_mask`.
- Rename column: `score` → `labels` (required by `Trainer`).

### 3. Validation Set Construction
```python
dds = tok_ds.train_test_split(0.25, seed=42)
```
25% random split. Notebook includes an extended tutorial on underfitting/overfitting using polynomial regression and synthetic data, motivating why validation sets are necessary.

### 4. Test Set Preparation
```python
eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)
```

### 5. Pearson Correlation Metric
```python
def corr(x, y): return np.corrcoef(x, y)[0][1]
def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}
```
Used as the evaluation metric in `Trainer`. Notebook includes illustrated examples of what different Pearson r values look like using the California Housing dataset.

### 6. Training Arguments
```python
bs = 128; epochs = 4; lr = 8e-5

args = TrainingArguments(
    'outputs',
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    fp16=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to='none'
)
```

### 7. Model Creation & Training
```python
model = AutoModelForSequenceClassification.from_pretrained(
    'microsoft/deberta-v3-small', num_labels=1   # regression: single output
)
trainer = Trainer(model, args,
                  train_dataset=dds['train'],
                  eval_dataset=dds['test'],
                  tokenizer=tokz,
                  compute_metrics=corr_d)
trainer.train()
```
`num_labels=1` configures the model for regression. Training logs Pearson correlation per epoch on the validation split.

### 8. Inference & Submission
```python
preds = trainer.predict(eval_ds).predictions.astype(float)
preds = np.clip(preds, 0, 1)   # clip out-of-range predictions

submission = datasets.Dataset.from_dict({'id': eval_ds['id'], 'score': preds})
submission.to_csv('submission.csv', index=False)
```
Clipping is required because the regression head can predict values outside `[0, 1]`.

---

## Output

| File | Description |
|---|---|
| `submission.csv` | `id` + `score` predictions clipped to `[0, 1]` |
| `outputs/` | Saved model checkpoints |

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| DeBERTa-v3-small | Strong NLP backbone; disentangled attention performs well on short text similarity |
| Input concatenation (`TEXT1/TEXT2/ANC1`) | Encodes all relevant fields as a single text, maximising cross-attention between anchor and target |
| `num_labels=1` | Frames as regression (not classification), appropriate for a continuous similarity score |
| `fp16=True` | Mixed precision for faster training on GPU |
| Cosine LR schedule + warmup | Standard transformer fine-tuning schedule |
| `np.clip(preds, 0, 1)` | Corrects invalid regression outputs before submission |

---

## Notable Educational Sections

The notebook contains extended tutorials (not strictly needed for the pipeline):

| Section | Topic Covered |
|---|---|
| Tokenization walkthrough | Subword tokenization, vocab lookup, `▁` prefix convention |
| Underfitting/overfitting | Polynomial regression demo with synthetic data |
| Validation set motivation | Why held-out data is necessary; train vs. test split distinction |
| Pearson correlation | Visual examples; correlation ≠ slope; outlier sensitivity |

---

## Suggested Improvements

- Use a larger DeBERTa variant (`deberta-v3-base` or `deberta-v3-large`).
- Switch to group-aware cross-validation (split by `anchor` to prevent leakage from the same anchor appearing in train and val).
- Add adversarial weight perturbation (AWP) — standard in top NLP competition solutions.
- Use multi-sample dropout on the classification head.
- Ensemble multiple seeds or model sizes.
- Incorporate CPC section descriptions as additional context (external patent taxonomy data).
