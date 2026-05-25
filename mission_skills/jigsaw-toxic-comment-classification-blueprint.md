# Blueprint: jigsaw-toxic-comment-classification-challenge.ipynb

## Overview

This notebook implements a **NBSVM (Naive Bayes – Support Vector Machine)** baseline for the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The approach uses TF-IDF features combined with Naive Bayes log-count ratios fed into Logistic Regression — a method introduced in the paper *"Baselines and Bigrams: Simple, Good Sentiment and Topic Classification"* (Wang & Manning, 2012).

---

## Purpose

Build a strong, lightweight NLP baseline that classifies Wikipedia comments into 6 toxicity labels simultaneously using classical ML (no neural networks required).

---

## Dependencies

| Library | Role |
|---|---|
| `pandas`, `numpy` | Data loading and manipulation |
| `sklearn.linear_model.LogisticRegression` | Classifier |
| `sklearn.feature_extraction.text.TfidfVectorizer` | Bag-of-words feature extraction |
| `re`, `string` | Custom tokenization |

---

## Competition Details

| Field | Value |
|---|---|
| Task | Multi-label binary classification |
| Input | Free-text comment strings |
| Labels | `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` |
| Evaluation | Mean column-wise ROC AUC |
| Data files | `train.csv`, `test.csv`, `sample_submission.csv` |

---

## Data

- **`train.csv`**: columns `id`, `comment_text`, plus 6 binary label columns.
- **`test.csv`**: columns `id`, `comment_text` — no labels.
- A synthetic `none` column is added to flag comments with no toxicity labels at all.
- Empty/null comments are filled with the string `"unknown"` before vectorization.

---

## Pipeline

### 1. Data Loading & Exploration
```python
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
```
- Inspect comment length distribution (`mean`, `std`, `max`, histogram).
- Add `none` column: `1 - max(label_cols, axis=1)`.
- Fill NaN comments with `"unknown"`.

### 2. Custom Tokenizer
```python
re_tok = re.compile(f'([{string.punctuation}""¨«»®´·º½¾¿¡§£₤''])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
```
Splits punctuation into separate tokens to improve n-gram coverage.

### 3. TF-IDF Vectorization
```python
vec = TfidfVectorizer(
    ngram_range=(1,2), tokenizer=tokenize,
    min_df=3, max_df=0.9,
    strip_accents='unicode',
    use_idf=1, smooth_idf=1, sublinear_tf=1
)
trn_term_doc = vec.fit_transform(train['comment_text'])
test_term_doc = vec.transform(test['comment_text'])
```
Key settings: unigrams + bigrams, sublinear TF scaling, IDF smoothing, Unicode accent stripping, minimum document frequency of 3.

### 4. Naive Bayes Log-Count Ratio
```python
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
```
Computes the Naive Bayes prior ratio `r = log(P(feature|positive) / P(feature|negative))` with add-1 (Laplace) smoothing.

### 5. Per-Label NBSVM Model
```python
def get_mdl(y):
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)          # scale TF-IDF by NB log-ratio
    return m.fit(x_nb, y), r
```
One binary model is trained per label (6 total). The TF-IDF matrix is element-wise multiplied by `r` before fitting.

### 6. Inference & Submission
```python
preds = np.zeros((len(test), len(label_cols)))
for i, j in enumerate(label_cols):
    m, r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]

submission = pd.concat([pd.DataFrame({'id': subm['id']}),
                        pd.DataFrame(preds, columns=label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)
```

---

## Output

| File | Description |
|---|---|
| `submission.csv` | Comment `id` + probability columns for each of the 6 labels |

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| TF-IDF over binary counts | Empirically improves LB score from ~0.59 → ~0.55 AUC |
| Bigrams `(1,2)` | Captures phrase-level toxicity signals |
| Separate model per label | Labels are not mutually exclusive; independent models work well |
| `C=4` regularization | Moderate regularization; `dual=True` for efficiency on sparse data |
| Sublinear TF | Dampens the effect of very frequent terms |

---

## Suggested Improvements

- Add character-level n-grams for robustness against deliberate misspellings.
- Ensemble NBSVM predictions with a neural model (e.g., LSTM or BERT).
- Tune regularization `C` per label via cross-validation.
- Handle class imbalance with `class_weight='balanced'` or upsampling.
- Use cross-validated OOF predictions instead of full-train fitting to reduce overfitting.
