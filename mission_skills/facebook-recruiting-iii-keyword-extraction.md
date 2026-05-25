1. Problem Understanding

Task type

* Multi-label text classification
* Predict multiple StackOverflow-style tags for each post
* Extremely large-scale sparse NLP classification problem

Evaluation metric

* Mean F1 score across predicted tags
* Prediction quality depends heavily on:
    * selecting correct tag count
    * balancing precision vs recall
    * handling rare labels

Key challenges

* Massive dataset size (millions of rows)
* Very high-cardinality label space (~10K target tags)
* Sparse TF-IDF matrices causing RAM pressure
* Duplicate leakage between train/test
* Long noisy HTML/code-heavy text
* Need efficient incremental training/inference

⸻

2. Data Pipeline (Code-Oriented)

load_data()

Responsibilities:

* Read train/test CSV files using chunked loading
* Avoid full-memory loading
* Load:
    * post text
    * tags
    * ids

Implementation decisions:

* Use generators or chunk iterators
* Prefer dtype=str
* Store intermediate sparse matrices on disk using:
    * scipy.sparse.save_npz
    * pickle/joblib

⸻

preprocess(text)

Text cleaning pipeline:

Remove:

* code blocks
* HTML tags
* URLs
* hyperlinks
* punctuation
* line breaks

Normalize:

* lowercase conversion
* whitespace cleanup

Suggested implementation

Use regex:

re.sub(...)

Important cleaning stages:

1. remove <code>...</code>
2. remove HTML
3. remove URLs
4. remove punctuation
5. lowercase

⸻

extract_meta_features(raw_text, clean_text)

Create numeric auxiliary features.

Raw-text features

* raw character length
* count of code blocks
* count of <a href
* count of "http"
* count of ">" characters

Clean-text features

* token count
* cleaned text length

Scaling

Apply MinMaxScaler:

MinMaxScaler(feature_range=(0,1))

Output:

numpy.ndarray

⸻

detect_duplicates()

Purpose:

* exploit train/test leakage
* reduce duplicated training rows

Logic:

1. Hash cleaned text
2. Compare hashes across train/test
3. For duplicates:
    * directly copy known tags
4. If multiple labels exist:
    * union all labels

Training cleanup:

* remove exact duplicate train rows
* retain duplicates with different labels

Recommended hashing:

hashlib.md5(clean_text.encode()).hexdigest()

⸻

build_tfidf()

Vectorizer

Use:

TfidfVectorizer

Recommended params:

TfidfVectorizer(
    stop_words='english',
    max_features=20000,
    ngram_range=(1,1),
    min_df=2,
    sublinear_tf=True
)

Important decisions

* unigram-only
* no stemming
* no LSA/SVD
* sparse representation only

Memory handling

Process train data in chunks:

* 500K rows per chunk
* persist sparse matrices to disk

Final feature matrix

Concatenate:

scipy.sparse.hstack([tfidf_matrix, meta_features])

⸻

split_folds()

Validation approach:

* Use one chunk as validation
* Remaining chunks for training

Alternative implementation:

MultilabelStratifiedKFold

But original solution used:

* chunk-based validation split

⸻

3. Model Design

build_model()

Core model

One-vs-rest linear classifiers

Preferred classifier

SGDClassifier

Best-performing configuration

SGDClassifier(
    loss='modified_huber',
    penalty='l2',
    alpha=1e-5,
    max_iter=20,
    random_state=seed,
    n_jobs=-1
)

Why:

* scalable
* sparse-friendly
* fast
* produces confidence scores

⸻

Multi-label setup

Approach

Train independent binary classifier for each tag:

tag_i -> positive/negative

Label selection

* Keep top 10K most frequent tags
* Ignore ultra-rare tags

⸻

Storage strategy

Since 10K models cannot remain in RAM:

* train in batches
* save models incrementally

Recommended:

joblib.dump(model)

Batch size:

1000 models per batch

⸻

4. Training Strategy

train_one_batch()

Workflow:

1. Load sparse feature chunk
2. Build binary targets
3. Train 1000 classifiers
4. Save models to disk

Pseudo-flow:

for tag in batch_tags:
    y = build_binary_target(tag)
    clf.fit(X_train, y)
    save_model(clf)

⸻

Loss function

loss='modified_huber'

Reason:

* better leaderboard performance
* robust confidence outputs

⸻

Optimization strategy

Linear SGD optimization:

* sparse-compatible
* online-friendly

No GPU required.

⸻

Memory optimization tricks

Important

* avoid dense arrays
* avoid pandas-heavy transforms
* persist intermediates to disk

Use:

csr_matrix

⸻

5. Validation Strategy

Cross-validation logic

Original approach:

* reserve first chunk as validation
* train on another chunk

Validation objectives:

* threshold tuning
* top-k tuning
* classifier comparison

⸻

OOF generation

Generate:

decision_function(X_val)

Store:

* sparse confidence matrix
* per-tag probabilities

Used for:

* threshold search
* ensemble experiments

⸻

6. Inference Pipeline

predict()

Per-batch inference

For each 1000-model batch:

1. Load models
2. Predict confidence scores
3. Save intermediate predictions

⸻

First-stage filtering

For each model batch:

select tags where score > 0.20

Rules:

* max 5 tags
* if none selected:
    * keep highest-scoring tag

Purpose:

* preserve rare tags

⸻

Global aggregation

Combine all selected tags across batches.

Second-stage filtering:

score > 0.10

Rules:

* max 5 tags
* fallback to best tag

⸻

post_process()

Duplicate override

If test sample exists in train:

use known tags directly

This is extremely high-impact.

⸻

7. Key Tricks (ACTIONABLE)

Trick 1 — Remove code blocks aggressively

If text contains:

<code>...</code>

Then:

remove entirely

Reason:

* reduces vocabulary explosion
* improves generalization

⸻

Trick 2 — Use unigram TF-IDF only

If considering bigrams:

DON'T USE

Reason:

* hurts leaderboard score
* increases RAM dramatically

⸻

Trick 3 — Restrict vocabulary

Set:

max_features=20000

Reason:

* prevents OOM
* improves training stability

⸻

Trick 4 — Train only top-frequency tags

Use:

top_10000_tags

Reason:

* rare tags underrepresented
* reduces wasted classifiers

⸻

Trick 5 — Use chunk-based processing

If sparse matrix too large:

split into 500K-row chunks

⸻

Trick 6 — Hybrid thresholding

Two-stage thresholding:

Stage 1:

score > 0.20

Stage 2:

score > 0.10

This improved F1 over:

* fixed top-k
* single-threshold approaches

⸻

Trick 7 — Duplicate leakage exploitation

If identical cleaned text exists:

copy union of known tags

Very large leaderboard gain.

⸻

Trick 8 — Keep sparse matrices throughout

Never convert:

csr_matrix -> dense

⸻

8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

import re
import gc
import hashlib
import joblib
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix, hstack, vstack, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.linear_model import SGDClassifier
# =========================================================
# CONFIG
# =========================================================
class CFG:
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    MAX_FEATURES = 20000
    TOP_K_TAGS = 10000
    CHUNK_SIZE = 500000
    MODEL_BATCH_SIZE = 1000
    STAGE1_THRESH = 0.20
    STAGE2_THRESH = 0.10
    MAX_TAGS = 5
    RANDOM_SEED = 42
# =========================================================
# UTILS
# =========================================================
def seed_everything(seed):
    pass
def save_pickle(obj, path):
    pass
def load_pickle(path):
    pass
# =========================================================
# DATA LOADING
# =========================================================
def load_data():
    """
    Load train/test data with chunking.
    """
    pass
# =========================================================
# TEXT CLEANING
# =========================================================
def remove_code_blocks(text):
    pass
def clean_html(text):
    pass
def remove_urls(text):
    pass
def preprocess_text(text):
    """
    Full cleaning pipeline.
    """
    pass
# =========================================================
# META FEATURES
# =========================================================
def extract_meta_features(raw_text, clean_text):
    """
    Create numeric metadata features.
    """
    pass
def build_meta_matrix(df):
    pass
# =========================================================
# DUPLICATE HANDLING
# =========================================================
def hash_text(text):
    pass
def detect_duplicates(train_df, test_df):
    """
    Detect overlapping posts.
    """
    pass
def remove_train_duplicates(train_df):
    pass
# =========================================================
# TF-IDF
# =========================================================
def build_vectorizer():
    pass
def fit_tfidf(train_texts):
    pass
def transform_tfidf(vectorizer, texts):
    pass
# =========================================================
# FEATURE PIPELINE
# =========================================================
def build_feature_matrix(tfidf_matrix, meta_matrix):
    """
    Combine sparse tfidf + meta features.
    """
    pass
# =========================================================
# LABEL PROCESSING
# =========================================================
def extract_top_tags(train_df):
    pass
def build_binary_targets(train_tags, target_tag):
    pass
# =========================================================
# MODELING
# =========================================================
def build_model():
    """
    SGDClassifier(modified_huber)
    """
    pass
def train_single_model(X_train, y_train):
    pass
def train_model_batch(X_train, train_tags, tag_batch):
    """
    Train 1000 binary classifiers.
    """
    pass
# =========================================================
# VALIDATION
# =========================================================
def validate_batch(models, X_val):
    pass
def tune_thresholds(val_scores, val_targets):
    pass
# =========================================================
# INFERENCE
# =========================================================
def predict_batch(models, X_test):
    pass
def stage1_filter(batch_scores, tag_names):
    """
    score > 0.20
    """
    pass
def stage2_filter(all_scores):
    """
    score > 0.10
    """
    pass
def merge_predictions(batch_predictions):
    pass
# =========================================================
# SUBMISSION
# =========================================================
def create_submission(test_ids, predictions):
    pass
# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    seed_everything(CFG.RANDOM_SEED)
    # -----------------------------
    # Load data
    # -----------------------------
    train_df, test_df = load_data()
    # -----------------------------
    # Preprocess text
    # -----------------------------
    train_df["clean_text"] = train_df["text"].apply(preprocess_text)
    test_df["clean_text"] = test_df["text"].apply(preprocess_text)
    # -----------------------------
    # Duplicate handling
    # -----------------------------
    duplicate_map = detect_duplicates(train_df, test_df)
    train_df = remove_train_duplicates(train_df)
    # -----------------------------
    # Meta features
    # -----------------------------
    train_meta = build_meta_matrix(train_df)
    test_meta = build_meta_matrix(test_df)
    # -----------------------------
    # TF-IDF
    # -----------------------------
    vectorizer = fit_tfidf(train_df["clean_text"])
    X_train_tfidf = transform_tfidf(
        vectorizer,
        train_df["clean_text"]
    )
    X_test_tfidf = transform_tfidf(
        vectorizer,
        test_df["clean_text"]
    )
    # -----------------------------
    # Final sparse matrices
    # -----------------------------
    X_train = build_feature_matrix(
        X_train_tfidf,
        train_meta
    )
    X_test = build_feature_matrix(
        X_test_tfidf,
        test_meta
    )
    # -----------------------------
    # Tag selection
    # -----------------------------
    top_tags = extract_top_tags(train_df)
    # -----------------------------
    # Train in batches
    # -----------------------------
    all_models = {}
    for i in range(0, len(top_tags), CFG.MODEL_BATCH_SIZE):
        tag_batch = top_tags[i:i+CFG.MODEL_BATCH_SIZE]
        models = train_model_batch(
            X_train,
            train_df["tags"],
            tag_batch
        )
        save_pickle(models, f"models_{i}.pkl")
    # -----------------------------
    # Inference
    # -----------------------------
    batch_predictions = []
    for i in range(0, len(top_tags), CFG.MODEL_BATCH_SIZE):
        models = load_pickle(f"models_{i}.pkl")
        preds = predict_batch(models, X_test)
        filtered = stage1_filter(
            preds,
            list(models.keys())
        )
        batch_predictions.append(filtered)
    final_predictions = merge_predictions(batch_predictions)
    final_predictions = stage2_filter(
        final_predictions
    )
    # -----------------------------
    # Duplicate override
    # -----------------------------
    final_predictions = apply_duplicate_overrides(
        final_predictions,
        duplicate_map
    )
    # -----------------------------
    # Save submission
    # -----------------------------
    create_submission(
        test_df["id"],
        final_predictions
    )
if __name__ == "__main__":
    main()

⸻

Function Explanations

Function	Purpose
preprocess_text()	Clean noisy HTML/code-heavy text
extract_meta_features()	Generate auxiliary numeric signals
detect_duplicates()	Leak exploitation via hash matching
fit_tfidf()	Build sparse unigram TF-IDF
build_feature_matrix()	Combine sparse text + metadata
train_model_batch()	Train 1000 OVR classifiers
predict_batch()	Generate decision scores
stage1_filter()	Rare-tag preserving prefilter
stage2_filter()	Final tag selection
create_submission()	Generate Kaggle CSV

⸻

9. Strategy Priority (IMPORTANT)

1. Most impactful techniques

A. Duplicate leakage exploitation

Largest single gain.

B. Cleaned-text unigram TF-IDF

Core predictive signal.

C. One-vs-rest SGD classifiers

Efficient and scalable.

D. Two-stage thresholding

Major F1 improvement.

⸻

2. Secondary improvements

A. Meta features

Small but consistent gain.

B. Duplicate removal in train

Reduces noise.

C. Confidence-based filtering

Better than fixed top-k.

⸻

3. Minor tricks

A. Chunk-based processing

Engineering necessity.

B. Sparse matrix persistence

RAM optimization.

C. Ensemble across chunks

Minimal improvement observed.

D. LSA/SVD attempts

Generally harmful for leaderboard score.