# Blueprint: Playground Series S3E18 — Enzyme Commission Classification

> ⚠️ **Nature of this notebook: Full end-to-end training and inference pipeline.**
> Feature engineering, model training (7-fold × 2 seeds), Optuna-weighted ensemble, SHAP analysis, and submission generation are all executed. RFECV feature selection and meta-prediction cells are present but serve as exploratory tools — the meta-prediction block is **commented out** and must not be re-enabled without understanding its overfitting behavior.
> Source file: `playground-series-s3e18.ipynb`

---

## Competition Details

| Field | Value |
|---|---|
| Competition | [Playground Series Season 3, Episode 18](https://www.kaggle.com/competitions/playground-series-s3e18) |
| Task | Multi-label binary classification: predict EC1 and EC2 enzyme commission flags independently |
| Input | RDKit molecular descriptor features (31 numerical + 1 categorical) |
| Targets | `EC1`, `EC2` — independent binary labels (0 or 1 per molecule) |
| Evaluation | Mean ROC-AUC of EC1 and EC2: `(AUC_EC1 + AUC_EC2) / 2` |

---

## Dependencies

| Library | Role |
|---|---|
| `xgboost` | XGBoost gradient boosting classifier |
| `lightgbm` | LightGBM gradient boosting classifier |
| `catboost` | CatBoost gradient boosting classifier with native categorical support |
| `sklearn` | HistGradientBoosting, RandomForest, LogisticRegression, ExtraTrees, AdaBoost, MLP, RFECV, StratifiedKFold |
| `optuna` | Hyperparameter search framework; `CmaEsSampler` used for ensemble weight optimization |
| `shap` | Feature importance visualization (TreeExplainer for XGBoost and CatBoost) |
| `category_encoders` | OrdinalEncoder, CountEncoder, CatBoostEncoder, OneHotEncoder (available but categorical encoding via `cat_cols` passed to model APIs directly in final pipeline) |
| `scipy.cluster.hierarchy` | Hierarchical clustering for EDA |
| `pandas`, `numpy` | Data manipulation |

**No GPU required.** `device='cpu'` is fixed throughout.

---

## Data

| File / Directory | Description |
|---|---|
| `/kaggle/input/playground-series-s3e18/train.csv` | Synthetic training data; 31 RDKit molecular descriptors + EC1–EC6 binary labels |
| `/kaggle/input/playground-series-s3e18/test.csv` | Test data; same 31 descriptor columns, no labels |
| `/kaggle/input/playground-series-s3e18/sample_submission.csv` | Template: `id`, `EC1`, `EC2` columns |
| `/kaggle/input/ec-mixed-class/mixed_desc.csv` | **External real data** — original (non-synthetic) molecular descriptors with EC1–EC6 labels encoded as `EC1_EC2_EC3_EC4_EC5_EC6` string in one column |

**External data preprocessing:**
```python
df = original['EC1_EC2_EC3_EC4_EC5_EC6'].str.split('_').reset_index()
df_expanded = pd.DataFrame(df['EC1_EC2_EC3_EC4_EC5_EC6'].tolist(),
                           columns=[f'EC{i+1}' for i in range(6)])
df_expanded['CIDs'] = df['CIDs']
df_expanded.set_index('CIDs', inplace=True)
original = pd.concat([original[df_test.columns], df_expanded.astype(int)], axis=1)
```
The combined training set is:
```python
train = pd.concat([df_train, original]).drop_duplicates()
```
- `df_train['is_generated'] = 1`, `original['is_generated'] = 0` — provenance flag added but then **dropped** before modeling.

---

## Feature Columns

**31 base numerical columns:**
`BertzCT`, `Chi1`, `Chi1n`, `Chi1v`, `Chi2n`, `Chi2v`, `Chi3v`, `Chi4n`, `EState_VSA1`, `EState_VSA2`, `ExactMolWt`, `FpDensityMorgan1`, `FpDensityMorgan2`, `FpDensityMorgan3`, `HallKierAlpha`, `HeavyAtomMolWt`, `Kappa3`, `MaxAbsEStateIndex`, `MinEStateIndex`, `NumHeteroatoms`, `PEOE_VSA10`, `PEOE_VSA14`, `PEOE_VSA6`, `PEOE_VSA7`, `PEOE_VSA8`, `SMR_VSA10`, `SMR_VSA5`, `SlogP_VSA3`, `VSA_EState9`, `fr_COO`, `fr_COO2`

**Categorical column:** `fr_COO` — passed to LightGBM as `categorical_feature`, to CatBoost via `cat_features`, to HistGradientBoosting via `categorical_features`.

**Dropped columns before modeling:**
```python
drop_cols = ['is_generated', 'fr_COO2']
binary_cols = ['EC3', 'EC4', 'EC5', 'EC6']  # not used as features
target_cols = ['EC1', 'EC2']                 # predicted separately
```
`fr_COO2` is dropped due to high correlation with `fr_COO`.

---

## Execution Constraints

The following parameters are **FIXED** and must not be changed without re-running Optuna hyperparameter search.

| Parameter | Fixed Value | Do NOT change to |
|---|---|---|
| `kfold` | `'skf'` (StratifiedKFold) | plain KFold — class balance must be preserved per fold |
| `n_splits` | `7` | `5` or `10` — changes fold count and OOF averaging denominator |
| `n_reapts` | `2` | `1` — reduces ensemble diversity; `3+` increases runtime significantly |
| `random_state` | `42` | any other value — changes all random seeds downstream |
| `n_estimators` | `9999` | lower values — early stopping controls actual iteration count |
| `early_stopping_rounds` | `400` (XGB/CAT), `800` (LGB — 2×) | lower values cause premature stopping |
| `n_trials` (OptunaWeights) | `5000` | lower values — ensemble weights may not converge |
| `device` | `'cpu'` | `'gpu'` without also setting `tree_method='gpu_hist'` and `predictor='gpu_predictor'` for XGBoost |
| `cat_cols` | `['fr_COO']` | adding `fr_COO2` or other columns — must re-validate categorical handling |
| `drop_cols` | `['is_generated', 'fr_COO2']` | removing `fr_COO2` from drops — reintroduces correlated feature |
| External data source | `ec-mixed-class/mixed_desc.csv` | any other path or file |
| Meta-prediction block | **Commented out** | Do NOT uncomment — found to overfit (CV improves, LB decreases) |

---

## Pipeline

### Step 1 — Data Loading and Provenance Tagging

```python
df_train = pd.read_csv(os.path.join(filepath, 'train.csv'), index_col=[0])
df_test  = pd.read_csv(os.path.join(filepath, 'test.csv'),  index_col=[0])
original = pd.read_csv(os.path.join(generated_filepath, 'mixed_desc.csv'), index_col=[0])

df_train['is_generated'] = 1
df_test['is_generated']  = 1
original['is_generated'] = 0
```

### Step 2 — External Data Label Parsing

The `original` dataset encodes all 6 EC labels as a single underscore-delimited string column. This is parsed into 6 separate binary columns and concatenated with the descriptor features:
```python
df = original['EC1_EC2_EC3_EC4_EC5_EC6'].str.split('_').reset_index()
df_expanded = pd.DataFrame(df['EC1_EC2_EC3_EC4_EC5_EC6'].tolist(),
                           columns=[f'EC{i+1}' for i in range(6)])
original = pd.concat([original[df_test.columns], df_expanded.astype(int)], axis=1)
```
Combined training set (deduplication prevents data overlap):
```python
train = pd.concat([df_train, original]).drop_duplicates()
```

### Step 3 — Feature Engineering

Applied per target label loop (`for target_col in ['EC1', 'EC2']`):

**a) Ratio and Product Features** (`create_features()`):
Pairwise ratio/product interactions between high-importance RDKit features:
```python
'BertzCT_MaxAbsEStateIndex_Ratio': df['BertzCT'] / (df['MaxAbsEStateIndex'] + 1e-12),
'BertzCT_ExactMolWt_Product':       df['BertzCT'] * df['ExactMolWt'],
# ... and ~10 more ratio/product pairs
```
`1e-12` epsilon prevents division by zero — must be preserved in all ratio features.

**b) Aggregate Features** (`AggFeatureExtractor`):
For each grouping column in `group_cols`, compute `mean` and `std` of `agg_col` numerical features:
```python
group_cols = [['EState_VSA2'], ['HallKierAlpha'], ['NumHeteroatoms'],
              ['PEOE_VSA10'], ['PEOE_VSA14'], ['PEOE_VSA6'], ['PEOE_VSA7'], ['PEOE_VSA8'],
              ['SMR_VSA10'], ['SMR_VSA5'], ['SlogP_VSA3'], ['fr_COO']]
```
Aggregation statistics are fit on the **combined** train+test set to avoid distribution shift:
```python
agg_extractor.fit(pd.concat([X_train, X_test], axis=0))
```

**c) Dropped columns:**
```python
drop_cols = ['is_generated', 'fr_COO2']
X_train = X_train.drop(drop_cols, axis=1)
X_test  = X_test.drop(drop_cols, axis=1)
```

### Step 4 — Cross-Validation and Model Training

**Configuration:**
```python
kfold              = 'skf'    # StratifiedKFold
n_splits           = 7
n_reapts           = 2        # 2 random seeds
random_state       = 42
n_estimators       = 9999     # early stopping controls actual iterations
early_stopping_rounds = 400   # 800 for LightGBM
n_trials           = 5000     # Optuna ensemble weight search
device             = 'cpu'
```

**Random seed list** is sampled reproducibly:
```python
random.seed(42)
random_state_list = random.sample(range(9999), 2)
```
This produces 2 specific seeds. The exact seeds depend on Python's random state at this point — do not change `random.seed(42)` or `n_reapts`.

**Per-fold per-label training loop:**
For each of `7 folds × 2 seeds = 14 iterations` per target label:
1. Fit all models in `Classifier(target_col, n_estimators, device, seed)`.
2. Collect OOF predictions from each model.
3. Run `OptunaWeights` to find the best weighted combination (CMAsampler, 5000 trials).
4. Accumulate test predictions: `test_predss += optweights.predict(test_preds) / (7 * 2)`.

### Step 5 — Model Zoo

**EC1 active models** (7 total):

| Key | Model | Key Hyperparameters |
|---|---|---|
| `xgb` | XGBClassifier | lr=0.00765, max_depth=9, subsample=0.955, colsample=0.303, grow_policy=lossguide |
| `lgb` | LGBMClassifier | lr=0.02280, num_leaves=41, colsample=0.443, subsample=0.751, min_child=47 |
| `cat` | CatBoostClassifier | lr=0.02026, depth=3, grow_policy=Depthwise, bagging_temp=0.478 |
| `lgb2` | LGBMClassifier | lr=0.01503, num_leaves=122, colsample=0.577, subsample=0.415, min_child=89 |
| `hgb` | HistGradientBoostingClassifier | lr=0.0366, max_depth=30, max_leaf_nodes=12, min_samples_leaf=52 |
| `rfc` | RandomForestClassifier | n_estimators=500 |
| `lrc` | Pipeline(StandardScaler + LogisticRegression) | max_iter=500 |
| `mlp` | Pipeline(StandardScaler + MLPClassifier) | hidden=(100,), max_iter=800, early_stopping=True |
| `ada` | AdaBoostClassifier | n_estimators=100 |

**EC2 active models** (12 total — all EC1 models plus):

| Key | Model | Key Hyperparameters |
|---|---|---|
| `xgb2` | XGBClassifier | lr=0.03046, max_depth=3, subsample=0.355, colsample=0.741, grow_policy=depthwise |
| `cat2` | CatBoostClassifier | lr=0.0992, depth=5, grow_policy=Lossguide, bagging_temp=9.036 |
| `etc` | ExtraTreesClassifier | n_estimators=800 |

**Critical constraint — EC1 vs EC2 model sets are different:**
- EC1 does not include `xgb2`, `cat2`, `etc` (these are commented out or absent from EC1's model dict).
- EC2 includes all 12 models. The `Classifier._define_model(target_col)` method is the single source of truth — do not add EC2's extra models to EC1 without Optuna re-tuning.

**Early stopping by framework:**
```python
# LightGBM — 2× rounds
early_stopping_rounds_ = int(early_stopping_rounds * 2)  # = 800
model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)],
          categorical_feature=cat_cols,
          early_stopping_rounds=early_stopping_rounds_, verbose=verbose)

# CatBoost — uses Pool with cat_features
model.fit(Pool(X_train_, y_train_, cat_features=cat_cols),
          eval_set=Pool(X_val, y_val, cat_features=cat_cols),
          early_stopping_rounds=early_stopping_rounds_, verbose=verbose)

# XGBoost — standard
model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)],
          early_stopping_rounds=early_stopping_rounds_, verbose=verbose)

# Non-boosting models (rfc, lrc, mlp, ada, etc) — no early stopping
model.fit(X_train_, y_train_)
```

### Step 6 — Optuna Weighted Ensemble

```python
class OptunaWeights:
    def fit(self, y_true, y_preds):
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        self.study = optuna.create_study(sampler=sampler, direction='maximize')
        objective = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f'weight{n}']
                        for n in range(len(y_preds))]
```

- `CmaEsSampler` (Covariance Matrix Adaptation Evolution Strategy) is used for the weight search — not random TPE. This is more efficient for continuous weight optimization but requires `n_trials=5000` to converge reliably.
- Weights are normalized via `np.average(..., weights=weights)` — unnormalized floats in [1e-15, 1] are passed; numpy handles the relative weighting.
- One `OptunaWeights` instance is fit per fold per seed per target label — weights are not shared across folds.

### Step 7 — SHAP Analysis (Exploratory)

SHAP TreeExplainer is run on the **last fold's** XGBoost and CatBoost models for both EC1 and EC2:
```python
explainer = shap.TreeExplainer(model=trained_models_list[0]['xgb'][-1])
shap_values = explainer.shap_values(X=X_val_list[0])
shap.summary_plot(shap_values, X_val_list[0], plot_type="bar")
```
- `trained_models_list[0]` = EC1 trained models; `trained_models_list[1]` = EC2.
- Only GBDT models are SHAP-compatible; `rfc`, `mlp`, `ada`, `lrc` are excluded from SHAP analysis.
- SHAP results are **visualization only** — they do not feed back into training or feature selection.

### Step 8 — Submission

```python
def make_submission(target_cols, test_predss_list, prefix=''):
    sub = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
    for target_col, test_predss in zip(target_cols, test_predss_list):
        sub[f'{target_col}'] = test_predss
    sub.to_csv(f'{prefix}submission.csv', index=False)
    return sub

sub = make_submission(target_cols, test_predss_list, prefix='')
```

- `target_cols = ['EC1', 'EC2']`; `test_predss_list` is a list of two arrays aligned to `test.csv` row order.
- Output: `submission.csv` with columns `id`, `EC1`, `EC2` (float probabilities in [0, 1]).

---

## Output

| File | Description |
|---|---|
| `submission.csv` | `id`, `EC1` probability, `EC2` probability |

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| EC1 and EC2 trained independently | Labels have different predictive structure; separate Optuna-tuned hyperparameters per label produce better per-label AUC |
| EC1 uses 9 models, EC2 uses 12 | EC2 benefits from additional diversity (`xgb2`, `cat2`, `etc`); validated by per-label Optuna sweep |
| `fr_COO` as categorical, `fr_COO2` dropped | `fr_COO` encodes a discrete count feature; `fr_COO2` is highly correlated — including both inflates feature count without information gain |
| External `ec-mixed-class` data merged | Real enzyme data (non-synthetic) provides distribution anchor — improves generalization from synthetic Kaggle-generated training set |
| `pd.concat(...).drop_duplicates()` for merge | Removes exact duplicate rows between synthetic and real data to prevent double-counting |
| `AggFeatureExtractor` fit on train+test | Avoids target leakage in aggregation statistics — aggregation moments computed over the full feature distribution, not just training labels |
| Epsilon `1e-12` in ratio features | Prevents division by zero for molecular features that can be exactly 0 |
| Meta-prediction commented out | Appending EC1 predictions as a feature for EC2 training (and vice versa) improved CV AUC but reduced public leaderboard score — clear sign of overfitting to CV noise |
| `CmaEsSampler` in OptunaWeights | More efficient than TPE for continuous-weight optimization in high-dimensional weight space (9–12 models); requires 5000 trials to converge |
| LightGBM early stopping = 2× | LightGBM's internal stopping is evaluated less frequently than XGBoost/CatBoost; doubling patience compensates for this and prevents premature stopping |

---

## SOTA Gap

| Aspect | This Notebook | Competition SOTA |
|---|---|---|
| Feature engineering | Ratio/product pairs + group aggregates | ECFP fingerprints, FCFP fingerprints, Morgan circular features (additional external datasets) |
| Decomposition features | NMF available but commented out | PCA/UMAP/TSNE embeddings as supplementary features |
| Model diversity | 9–12 GBDT + sklearn models | Neural networks (TabNet, MLP with attention) + GBDT ensemble |
| Ensemble method | Optuna CMA-ES weighted average | Stacking with meta-learner; Bayesian model combination |
| Categorical encoding | Native cat feature in LGB/CatBoost | Target encoding, CatBoost encoding per fold |
| External data | `ec-mixed-class/mixed_desc.csv` only | Multiple BRENDA, ExPASy, ChEMBL molecular datasets |
| CV strategy | StratifiedKFold 7-fold × 2 seeds | GroupKFold by molecular scaffold to prevent similar-structure leakage |
| Approx. Mean AUC | ~0.86 (estimated from CV) | ~0.89+ (top solutions) |

---

## Suggested Improvements

1. **Add ECFP and FCFP fingerprint features** — the commented-out `mixed_ecfp.csv` and `mixed_fcfp.csv` files in `ec-mixed-class` provide Morgan circular fingerprints and feature-class fingerprints; these binary feature vectors encode structural patterns complementary to the RDKit descriptor features and typically add ~0.005–0.010 AUC.
2. **Enable NMF decomposition features** — the `add_decomp_features()` function is implemented and available; NMF components of the molecular descriptor matrix capture latent structural groupings; the code to call it is commented out in cell 15 with the note "if target_col == 'EC2'".
3. **Add GroupKFold by molecular scaffold** — molecules with similar scaffolds can appear in both train and val under StratifiedKFold, inflating CV AUC; scaffold-based GroupKFold (using RDKit `MurckoScaffold`) prevents this and produces more reliable fold estimates.
4. **Re-enable CatBoost encoding for `fr_COO`** — the `MyCategoryEncoders` with `encode='count'` is implemented and available but commented out; CatBoostEncoder applied per-fold is a strong alternative to native categorical handling.
5. **Add TabNet or MLP with feature embeddings** — the existing `mlp` model uses a single hidden layer (100 units) with StandardScaler; a deeper MLP (3–4 layers, 256–512 units) or TabNet with attention mechanisms typically improves AUC on molecular descriptor data.
6. **Calibrate predicted probabilities** — the ensemble outputs raw model probabilities that may not be well-calibrated; applying `CalibratedClassifierCV` (isotonic regression) per label post-hoc can tighten AUC by aligning probability scores with true label frequencies.
7. **Use RFECV results to eliminate low-importance features** — the RFECV cell (cell 22) is present and functional but uses `n_estimators=50` for speed; running it with full `n_estimators=9999` identifies which features to drop and can reduce overfitting from the expanded feature set after aggregation.
