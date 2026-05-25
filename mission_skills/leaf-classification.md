# Leaf Classification — Blueprint

## 1. Problem Understanding
- **Task:** Multi-class classification (plant species identification from leaf features).
- **Metric:** Multi-class Log Loss (primary); Accuracy (secondary).
- **Target:** 100 plant species classes, 990 training samples (~10 samples/class).
- **Key Challenges:** High class count relative to sample size (100 classes / 990 rows), pre-extracted tabular features (no raw images), no hyperparameter tuning by default, need stratified splitting to guarantee all classes appear in both train/val.

---

## 2. Data Pipeline
- **`load_data()`**: Read `train.csv` / `test.csv`. Columns: `id`, `species`, + 192 pre-extracted leaf feature columns.
- **`encode(train, test)`**:
  - `LabelEncoder().fit(train.species)` → integer labels; save `le.classes_` for submission column names.
  - Save `test.id` for submission index.
  - Drop `['species', 'id']` from train, `['id']` from test.
  - Returns: `train, labels, test, test_ids, classes`.
- **`preprocess()`**: Features are already normalized floats — no scaling strictly required, but `StandardScaler` recommended for distance-based and linear classifiers (KNN, SVM, LDA).

---

## 3. Model Design
- **Approach:** Tabular classifier showdown — 10 sklearn estimators evaluated out-of-the-box.
- **Classifier Pool:**

| Classifier | Notes |
|---|---|
| `KNeighborsClassifier(3)` | Distance-sensitive; benefits from scaling |
| `SVC(kernel='rbf', C=0.025, probability=True)` | Low C → underfit baseline |
| `NuSVC(probability=True)` | Nu-parameterized SVM |
| `DecisionTreeClassifier()` | High variance, interpretable |
| `RandomForestClassifier()` | Ensemble, robust baseline |
| `AdaBoostClassifier()` | Boosting on weak learners |
| `GradientBoostingClassifier()` | Strongest tree ensemble |
| `GaussianNB()` | Fast, assumes feature independence |
| `LinearDiscriminantAnalysis()` | Best performer on this dataset; used for final submission |
| `QuadraticDiscriminantAnalysis()` | More flexible than LDA; may overfit |

---

## 4. Training Strategy
- **`create_folds()`**: `StratifiedShuffleSplit(n_iter=10, test_size=0.2, random_state=23)` — stratification mandatory given 100 classes / 990 samples ratio.
- **Loop:** Fit each classifier on `X_train`, evaluate on `X_test` (last split used).
- **No hyperparameter tuning** in baseline — all defaults except `KNN(3)` and `SVC(C=0.025)`.
- **Logging:** Collect `(Classifier, Accuracy, Log Loss)` per model into a `pd.DataFrame` for comparison.

---

## 5. Validation Strategy
- **Split:** `StratifiedShuffleSplit` with `test_size=0.2`; last fold used for evaluation.
- **Metrics:**
  - `accuracy_score(y_test, clf.predict(X_test))`
  - `log_loss(y_test, clf.predict_proba(X_test))` ← competition primary metric
- **Visualization:** `sns.barplot` of Accuracy and Log Loss across all classifiers.

---

## 6. Inference Pipeline
- **`predict()`**: Refit chosen classifier (`LinearDiscriminantAnalysis`) on full `X_train`.
- **Output:** `clf.predict_proba(test)` → 100-column probability matrix.
- **`format_submission()`**: Wrap probabilities in `pd.DataFrame(columns=classes)`, prepend `id` column from `test_ids`.

---

## 7. Key Tricks (ACTIONABLE)
- **If** dataset has many classes and few samples → **do** use `StratifiedShuffleSplit`; random split risks missing classes in val.
- **If** comparing classifiers → **do** log both Accuracy and Log Loss; a model with high accuracy can still have poor probability calibration.
- **If** using SVM or KNN → **do** apply `StandardScaler` first; raw feature magnitudes hurt distance metrics.
- **If** `SVC` shows poor Log Loss → **do** tune `C` upward (baseline `C=0.025` is heavily regularized).
- **If** selecting final model → **do** prefer LDA for this dataset: it handles high class count with low sample size via class covariance pooling.
- **If** submitting → **do** use `predict_proba` (not `predict`); competition scores on probability distributions, not hard labels.

---

## 8. Code Structure

```python
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def load_data(train_csv, test_csv): pass            # read CSVs

def encode(train, test): pass                       # LabelEncode species → labels, classes, test_ids

def preprocess(X_train, X_test): pass               # optional StandardScaler

def create_folds(labels, n_iter=10, test_size=0.2): pass  # StratifiedShuffleSplit

def run_showdown(X_train, y_train, X_test, y_test): pass  # loop classifiers → log Accuracy + Log Loss

def plot_results(log): pass                         # sns.barplot for Accuracy and Log Loss

def predict_and_format(clf, X_train, y_train, test, test_ids, classes): pass  # refit → predict_proba → DataFrame

def main():
    train, test = load_data('train.csv', 'test.csv')
    train, labels, test, test_ids, classes = encode(train, test)

    sss = create_folds(labels)
    for train_idx, test_idx in sss:
        X_train, X_test = train.values[train_idx], train.values[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

    log = run_showdown(X_train, y_train, X_test, y_test)
    plot_results(log)

    # submission = predict_and_format(LinearDiscriminantAnalysis(), X_train, y_train, test, test_ids, classes)
    # submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
```

---

## 9. Strategy Priority

1. **High Impact:** Stratified splitting (mandatory for 100-class low-data regime) · LDA as final model (best log loss on this feature set) · `predict_proba` output for submission.
2. **Medium Impact:** `StandardScaler` for SVM/KNN · Log Loss as primary selection metric over Accuracy · Grid-search `C` for SVC.
3. **Minor:** Multi-iteration shuffle split for stable val estimate · Barplot visual comparison across classifiers · AdaBoost / GBM as ensemble alternatives worth tuning.