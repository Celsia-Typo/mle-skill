# Blueprint: Predict Volcanic Eruptions — INGV Seismic Data

## Overview

This notebook builds a **multi-model ensemble regression pipeline** to predict the time remaining before a volcanic eruption from 10-sensor seismic signal segments. It covers full data exploration, FFT-based feature engineering, and training of LightGBM, XGBoost, and a Keras neural network, with a weighted ensemble submission.

---

## Competition Details

| Field | Value |
|---|---|
| Competition | [Predict Volcanic Eruptions — INGV-OE](https://www.kaggle.com/competitions/predict-volcanic-eruptions-ingv-oe) |
| Task | Regression — predict `time_to_eruption` (seconds) from seismic sensor data |
| Input | 10 sensors × 60,001 time steps per segment (CSV files) |
| Target | `time_to_eruption` in seconds |
| Evaluation | RMSE |

---

## Dependencies

| Library | Role |
|---|---|
| `numpy`, `pandas` | Numerical computing and data handling |
| `glob` | File discovery |
| `plotly` | Interactive EDA visualizations |
| `sklearn` | Train/val split, RFE, metrics |
| `lightgbm` | LightGBM regressor |
| `xgboost` | XGBoost regressor |
| `optuna` | Hyperparameter optimization (LightGBM) |
| `tensorflow.keras` | Neural network regressor |

---

## Data Format

- `train.csv` — metadata: `segment_id`, `time_to_eruption`
- `train/{segment_id}.csv` — 10 sensor columns × 60,001 rows (pre-normalized within each segment)
- `test/{segment_id}.csv` — same structure, no labels
- ~4,000 train segments, ~7,000 test segments

---

## Pipeline

### 1. EDA — Target Distribution

- `time_to_eruption` histogram: right-skewed, median ≈ several months, high variance.
- Each segment has exactly 10 sensors and 60,001 observations (confirmed by iterating all files).
- **Missing sensors:** `sensor_4` and `sensor_6` have no fully-absent records in training. Several segments have partially missing sensors, tracked as `missed_percent_sensorX` features.

### 2. Missing Sensor Analysis

For each segment, the notebook records:
- `has_missed_sensors` (boolean)
- `missed_percent_sensorX` for each of the 10 sensors

These are merged into `train` as additional meta-features. Distribution of `time_to_eruption` is compared between segments with and without missing sensors.

### 3. Feature Engineering (`build_features`)

For each sensor signal, the following statistical and spectral features are computed:

```python
def build_features(signal, ts, sensor_id):
    f = np.fft.fft(signal)
    f_real = np.real(f)

    X[f'{sensor_id}_sum']       = signal.sum()
    X[f'{sensor_id}_mean']      = signal.mean()
    X[f'{sensor_id}_std']       = signal.std()
    X[f'{sensor_id}_var']       = signal.var()
    X[f'{sensor_id}_min']       = signal.min()
    X[f'{sensor_id}_max']       = signal.max()
    X[f'{sensor_id}_range']     = signal.max() - signal.min()
    X[f'{sensor_id}_maxtomin']  = signal.max() / signal.min()
    X[f'{sensor_id}_abs_mean']  = np.abs(signal).mean()
    X[f'{sensor_id}_abs_std']   = np.abs(signal).std()
    X[f'{sensor_id}_fft_real_mean'] = f_real.mean()
    X[f'{sensor_id}_fft_real_std']  = f_real.std()
    X[f'{sensor_id}_fft_real_max']  = f_real.max()
    X[f'{sensor_id}_fft_real_min']  = f_real.min()
    # ... additional FFT and signal stats
```

Result: ~200+ features per segment (10 sensors × ~20 features each), plus the missing-sensor meta-features.

### 4. Feature Selection

Two passes of feature reduction before tree models:
- **Low-correlation drop:** Remove features with `abs(corr(feature, target)) < 0.01`.
- **High inter-correlation drop:** Remove features with pairwise correlation > 0.98 (retains one from each correlated pair).

### 5. Train/Validation Split
```python
train, val, y, y_val = train_test_split(train, y, test_size=0.2, random_state=666, shuffle=True)
```

### 6. Models

#### Model 1 — LightGBM (baseline)
```python
lgb = LGBMRegressor(random_state=666, max_depth=7, n_estimators=250, learning_rate=0.12)
```

#### Model 2 — LightGBM (Optuna-tuned)
```python
# Optuna TPE search over: num_leaves, n_estimators, max_depth,
#                         min_child_samples, learning_rate, min_data_in_leaf
study = optuna.create_study(sampler=TPESampler(seed=666))
study.optimize(create_model, n_trials=100)
```
Best params: `num_leaves=31, n_estimators=138, max_depth=8, min_child_samples=182, lr=0.167`

#### Model 3 — XGBoost with RFE
```python
rfe_estimator = RFE(estimator=xgb, n_features_to_select=100)
pipe_lgb = Pipeline([('rfe', rfe_estimator), ('lgb', lgb_tuned)])
```

#### Model 4 — XGBoost (short)
```python
xgb_short = XGBRegressor(max_depth=6, n_estimators=189, learning_rate=0.099, gamma=0.788)
```
Trained on the reduced feature set.

#### Model 5 — Keras Neural Network (3-fold CV)
```python
# Log-transform target: yy = np.log1p(y)
model = Sequential([
    Dense(256, activation='relu', input_dim=X.shape[1]),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])
# Loss: custom RMSE
# 3-fold KFold, EarlyStopping(patience=10)
# Predictions averaged across 3 folds, then np.expm1() to undo log transform
```

### 7. Ensemble Submission

```python
test_set['time_to_eruption'] = (
    preds1 * 0.50 +   # LightGBM baseline
    preds2 * 0.05 +   # LightGBM + RFE pipeline
    preds3 * 0.30 +   # XGBoost + RFE pipeline
    preds4 * 0.05 +   # Neural network average
    preds5 * 0.10     # XGBoost short (reduced features)
)
```

LightGBM baseline receives the highest weight (0.5); neural network receives the lowest (0.05).

---

## Output

| File | Description |
|---|---|
| `submission.csv` | `segment_id` + `time_to_eruption` (predicted seconds to eruption) |

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| FFT features | Seismic signals have frequency-domain patterns; time-domain stats alone are insufficient |
| `log1p` target transform for NN | Reduces right skew; `expm1` applied at inference |
| Two-stage feature selection | Removes noise features and collinear duplicates to improve tree model performance |
| Weighted ensemble (0.5/0.05/0.3/0.05/0.1) | Reflects empirical validation RMSE of each model |
| Missing sensor tracking | Absence of certain sensors is correlated with specific eruption timing patterns |

---

## Suggested Improvements

- Use `GroupKFold` if multiple segments belong to the same volcanic sequence, to avoid temporal leakage.
- Add wavelet features (multi-resolution signal decomposition) alongside FFT.
- Use `log1p` target transform consistently across all models.
- Tune the ensemble weights using held-out OOF predictions via Nelder-Mead or Optuna.
- Consider LSTM/1D-CNN models operating directly on the raw signal sequences.
