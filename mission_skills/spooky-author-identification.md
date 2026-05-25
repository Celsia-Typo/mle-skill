# Approaching (Almost) Any NLP Problem on Kaggle — Blueprint

## 1. Problem Understanding
- **Task:** Multi-class text classification (author attribution: EAP / HPL / MWS).
- **Metric:** Multi-class Log Loss (`multiclass_logloss`), lower is better.
- **Target:** 3-class authorship prediction from raw sentence text.
- **Key Challenges:** Pure text input (no structured features), probabilistic output required (`predict_proba`), wide model family gap between classical sparse features and deep sequence models, ensembling across incompatible feature spaces.

---

## 2. Data Pipeline
- **`load_data()`**: Read `train.csv`, `test.csv`, `sample_submission.csv`. Columns: `id`, `text`, `author`.
- **`encode()`**: `LabelEncoder().fit_transform(train.author)` → integer labels `y`. Save `lbl_enc.classes_` for submission columns.
- **`split()`**: `train_test_split(text, y, test_size=0.1, stratify=y, random_state=42, shuffle=True)` → `xtrain, xvalid, ytrain, yvalid`.
- **Feature tracks** (run in parallel, feed different model families):

| Track | Function | Output Shape |
|---|---|---|
| TF-IDF | `TfidfVectorizer(ngram_range=(1,3), min_df=3, sublinear_tf=True)` fit on train+valid | sparse matrix |
| Count | `CountVectorizer(ngram_range=(1,3))` fit on train+valid | sparse matrix |
| SVD | `TruncatedSVD(n_components=120)` on TF-IDF → `StandardScaler` | dense (N, 120) |
| GloVe | `sent2vec()` mean-pool on `glove.840B.300d` → L2-normalize | dense (N, 300) |
| Sequence | Keras `Tokenizer` → `pad_sequences(maxlen=70)` + embedding matrix | (N, 70) int ids |

- **`sent2vec(s)`**: lowercase → `word_tokenize` → remove stopwords & non-alpha → lookup GloVe → sum → L2-normalize. Returns `np.zeros(300)` if no valid words found.

---

## 3. Model Design
- **Classical models** (on sparse/SVD features):

| Model | Feature | Notes |
|---|---|---|
| `LogisticRegression(C=1.0)` | TF-IDF / Count | Strong baseline |
| `MultinomialNB()` | TF-IDF / Count | Fast; tune `alpha` |
| `SVC(C=1.0, probability=True)` | SVD + Scaled | Slow; requires SVD+scaler pipeline |
| `XGBClassifier(max_depth=7, n_estimators=200, lr=0.1)` | TF-IDF / Count / SVD / GloVe | Dense features preferred |

- **Deep models** (on GloVe / sequence features):

| Model | Architecture |
|---|---|
| Dense NN | GloVe(300) → Dense(300, relu) → Dropout(0.2) → BN → Dense(300, relu) → Dropout(0.3) → BN → Dense(3, softmax) |
| LSTM | Embedding(GloVe, trainable=False) → SpatialDropout(0.3) → LSTM(300, dropout=0.3) → Dense(1024)×2 → Dense(3, softmax) |
| BiLSTM | Same as LSTM with `Bidirectional` wrapper |
| GRU | Embedding → SpatialDropout(0.3) → GRU(300)×2 (return_sequences) → Dense(1024)×2 → Dense(3, softmax) |

- **Embedding matrix**: `np.zeros((vocab_size+1, 300))`; fill from `embeddings_index` dict; leave OOV rows as zero.

---

## 4. Training Strategy
- **Classical:** `.fit(X_train, y_train)` → `.predict_proba(X_valid)` → log loss.
- **Deep (Keras):** `model.compile(loss='categorical_crossentropy', optimizer='adam')`. Labels → `np_utils.to_categorical`. `batch_size=64`, `epochs=5`. Add `EarlyStopping(patience=2)` on val loss.
- **Scale before neural nets:** `StandardScaler().fit_transform(xtrain_glove)` — mandatory before Dense NN.
- **Semi-supervised vectorizer fitting:** `tfv.fit(list(xtrain) + list(xvalid))` — include validation text in vocabulary construction (no label leakage since only vocab, not targets, is used).
- **`GridSearchCV` pipeline for classical tuning:**
  ```
  Pipeline([('svd', TruncatedSVD()), ('scl', StandardScaler()), ('lr', LogisticRegression())])
  param_grid = {'svd__n_components': [120, 180], 'lr__C': [0.1, 1.0, 10], 'lr__penalty': ['l1', 'l2']}
  scoring = make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)
  ```

---

## 5. Validation Strategy
- **Split:** Stratified 90/10 holdout (`stratify=y`) — stratification mandatory for 3-class balance.
- **Metric:** `multiclass_logloss(yvalid, predict_proba(xvalid))` (custom; clips predictions to `[1e-15, 1-1e-15]`).
- **OOF for ensembling:** `Ensembler` uses `StratifiedKFold(n_splits=3)` to generate level-0 OOF predictions as meta-features for level-1 model.

---

## 6. Inference Pipeline
- **`predict()`**: Call `clf.predict_proba(xtest)` for all models. Output shape: `(N_test, 3)`.
- **Ensembling — `Ensembler` class (stacking):**
  - **Level 0:** `[LR(TF-IDF), LR(Count), NB(TF-IDF, α=0.1), NB(Count)]` → OOF probability matrix `(N, 12)`.
  - **Level 1:** `XGBClassifier` trained on level-0 OOF matrix.
  - Each level trains per fold, saves OOF predictions; `predict()` refits on full data then scores test.
- **`format_submission()`**: `pd.DataFrame(predictions, columns=classes)` → insert `id` column → `.to_csv('submission.csv', index=False)`.

---

## 7. Key Tricks (ACTIONABLE)
- **If** building first model → **do** start with `TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True)` + `LogisticRegression`; it is the strongest single classical baseline.
- **If** using SVM → **do** apply `TruncatedSVD(n_components=120)` + `StandardScaler` first; raw sparse TF-IDF makes SVM infeasibly slow.
- **If** fitting vectorizers → **do** fit on `train + valid` text (semi-supervised); this expands vocabulary without leaking labels.
- **If** using GloVe sentence vectors → **do** L2-normalize the summed vector; unnormalized sums are dominated by long sentences.
- **If** training deep models on GloVe → **do** apply `StandardScaler` before Dense NN; skip scaling for LSTM/GRU (embedding handles it).
- **If** tuning NB → **do** grid-search `alpha` (`[0.001, 0.01, 0.1, 1, 10, 100]`); default `alpha=1.0` is rarely optimal and tuning yields ~8% log loss improvement.
- **If** ensembling → **do** use stacking (level-0 OOF → level-1 XGB) rather than simple averaging when model families are diverse (sparse vs. dense vs. sequence features).
- **If** submission requires probabilities → **do** always use `predict_proba`; hard-label `predict` output scores zero log loss credit.

---

## 8. Code Structure

```python
import numpy as np, pandas as pd
from sklearn import preprocessing, decomposition, pipeline, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, SpatialDropout1D, BatchNormalization
from keras.preprocessing import sequence, text
from keras.utils import np_utils

def load_data(train_csv, test_csv): pass                   # read CSVs

def encode(train): pass                                    # LabelEncoder → y, classes

def split(texts, y): pass                                  # stratified 90/10 holdout

def multiclass_logloss(actual, predicted, eps=1e-15): pass # custom log loss (clip + mean)

def build_tfidf(xtrain, xvalid): pass                      # fit on train+valid, transform both
def build_count(xtrain, xvalid): pass                      # same for CountVectorizer
def build_svd_scaled(xtrain_tfv, xvalid_tfv): pass         # TruncatedSVD(120) + StandardScaler

def load_glove(path): pass                                 # word → 300d vector dict
def sent2vec(s, embeddings_index): pass                    # tokenize → GloVe lookup → L2-norm
def build_glove_features(xtrain, xvalid): pass             # sent2vec → np.array

def build_sequence_features(xtrain, xvalid, embeddings_index): pass  # Tokenizer → pad → embedding matrix

def build_dense_model(): pass                              # 300→300→3 with BN + Dropout
def build_lstm_model(word_index, embedding_matrix): pass   # Embedding → SpatialDrop → LSTM → Dense×2 → 3
def build_bilstm_model(word_index, embedding_matrix): pass # Bidirectional wrapper variant
def build_gru_model(word_index, embedding_matrix): pass    # GRU×2 variant

def run_classical(X_train, y_train, X_valid, y_valid): pass  # loop classifiers → log loss table
def grid_search(X_train, y_train): pass                    # GridSearchCV with custom scorer

class Ensembler: pass                                      # multi-level stacking via StratifiedKFold OOF

def format_submission(predictions, test_ids, classes): pass  # DataFrame → submission.csv

def main():
    train, test = load_data('train.csv', 'test.csv')
    y, classes = encode(train)
    xtrain, xvalid, ytrain, yvalid = split(train.text.values, y)

    # Feature tracks
    xtrain_tfv, xvalid_tfv = build_tfidf(xtrain, xvalid)
    xtrain_ctv, xvalid_ctv = build_count(xtrain, xvalid)
    xtrain_svd, xvalid_svd = build_svd_scaled(xtrain_tfv, xvalid_tfv)
    xtrain_glove, xvalid_glove = build_glove_features(xtrain, xvalid)
    xtrain_seq, xvalid_seq, emb_matrix = build_sequence_features(xtrain, xvalid, load_glove('glove.840B.300d.txt'))

    # Classical baseline
    run_classical(xtrain_tfv, ytrain, xvalid_tfv, yvalid)

    # Deep models
    # model = build_lstm_model(word_index, emb_matrix); model.fit(xtrain_seq, ...)

    # Stacking ensemble
    # ens = Ensembler(model_dict, num_folds=3, ...); ens.fit(...); preds = ens.predict(...)

    # Submission
    # format_submission(preds, test_ids, classes)

if __name__ == "__main__":
    main()
```

---

## 9. Strategy Priority

1. **High Impact:** TF-IDF (ngram 1-3, sublinear_tf) + LogisticRegression baseline · GloVe `sent2vec` features for XGB/Dense · Stacking ensemble (LR+NB OOF → XGB meta-learner).
2. **Medium Impact:** Semi-supervised vectorizer fitting (train+valid vocab) · `GridSearchCV` on NB `alpha` and LR `C` · Bidirectional LSTM / dual-GRU for sequence modeling.
3. **Minor:** `TruncatedSVD(120)` + scaling before SVM · `SpatialDropout1D(0.3)` in recurrent models · `EarlyStopping` on val loss for Keras models.