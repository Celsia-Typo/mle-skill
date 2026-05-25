New York City Taxi Fare Prediction — Medal Solution Blueprint

1. Overall Strategy

Build a feature-engineered gradient boosting regression pipeline for NYC taxi fare prediction.

Core philosophy:

* Heavy feature engineering from coordinates + datetime
* Geographic priors (airports / landmarks)
* Distance-based handcrafted features
* Large-scale tabular boosting model
* Aggressive data cleaning
* Efficient memory handling for 20M+ rows

Primary model:

* LightGBM regression model (gbdt)

Target:

* Predict fare_amount

Evaluation metric:

* RMSE

⸻

2. Data Loading Strategy

Use Partial Loading

Training data is extremely large.

Blueprint rules:

* Load only a subset first (5M–25M rows)
* Use chunking if memory constrained
* Prefer:
    * nrows
    * explicit dtypes
    * garbage collection

Recommended:

pd.read_csv(..., nrows=20000000)

Memory optimization:

del unused_dataframe
gc.collect()

⸻

3. Mandatory Data Cleaning Rules

The model quality heavily depends on removing noisy samples.

Required Filters

Fare constraints

Keep only:

0 < fare_amount <= 500

Coordinate validity

Remove rows where any coordinate equals zero.

Required fields:

* pickup_longitude
* pickup_latitude
* dropoff_longitude
* dropoff_latitude

Optional strong filters

LLM should consider:

* NYC longitude bounds
* NYC latitude bounds
* passenger count validity
* remove extreme trip distances

Suggested geographic bounds:

longitude ∈ [-75, -72]
latitude ∈ [40, 42]

⸻

4. Core Geographic Features

4.1 Haversine Distance (MANDATORY)

This is the single most important feature.

LLM MUST implement great-circle distance.

Use vectorized NumPy implementation.

Formula concept:

d=2R\arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right)+\cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)

Feature name:

distance

Units:

* kilometers

Implementation requirements:

* fully vectorized
* radians conversion
* no Python loops

⸻

4.2 Bearing Feature (IMPORTANT)

Add directional information.

Feature:

bearing

Purpose:

* captures Manhattan directional structure
* helps model route orientation

Implementation:

atan2(...)

⸻

4.3 Coordinate Transformations

Convert coordinates to radians after raw distance calculations.

Required transformed features:

pickup_latitude_rad
pickup_longitude_rad
dropoff_latitude_rad
dropoff_longitude_rad

OR overwrite originals.

⸻

5. Landmark / Airport Distance Features (HIGH VALUE)

This solution class relies heavily on geographic priors.

Mandatory landmarks

Create distances to:

* JFK airport
* Newark airport
* LaGuardia airport
* Statue of Liberty
* NYC center

Each feature should represent:

pickup_to_landmark + dropoff_to_landmark

Example features:

jfk_dist
ewr_dist
lga_dist
sol_dist
nyc_dist

Why this works:

* airport trips have unique pricing behavior
* landmark traffic patterns differ
* fares correlate strongly with airport proximity

⸻

6. Datetime Features (MANDATORY)

Convert timestamp:

pickup_datetime

Extract:

Feature	Importance
hour	very high
weekday	high
month	medium
year	medium
day	low-medium

Recommended additions:

* weekend flag
* rush-hour flag
* night flag
* quarter
* weekofyear

⸻

7. Feature Engineering Extensions (Recommended)

LLM should attempt additional strong features.

Manhattan Approximation

abs(lat_diff) + abs(lon_diff)

Coordinate Differences

lat_diff
lon_diff

Interaction Features

Examples:

distance * passenger_count
hour * weekday

Cluster Features

Advanced medal-level option:

* KMeans pickup clusters
* KMeans dropoff clusters

⸻

8. Train / Validation Strategy

Split

Use:

train_test_split(...)

or preferably:

* KFold
* Time-aware split

Recommended:

test_size=0.1
random_state=123

⸻

9. Model Blueprint

Primary Model

Use:

* LightGBM Regressor

Objective:

regression

Metric:

rmse

⸻

Recommended Hyperparameters

Strong baseline:

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03-0.08,
    "num_leaves": 31-128,
    "max_depth": -1,
    "subsample": 0.7-0.9,
    "colsample_bytree": 0.5-0.8,
    "bagging_freq": 5-20,
    "min_child_samples": 10-100,
    "max_bin": 255-5000,
    "seed": 0
}

⸻

10. Early Stopping Rules

MANDATORY:

early_stopping_rounds

Recommended:

200-1000

Large boosting rounds:

10000+

Use best iteration for inference.

⸻

11. Categorical Features

Treat these as categorical:

year
month
day
weekday

Optional:

hour

⸻

12. Inference Pipeline Consistency (CRITICAL)

The feature engineering pipeline for test data MUST exactly match training.

LLM MUST:

* reuse all preprocessing functions
* reuse all landmark calculations
* preserve feature ordering
* avoid train/test feature mismatch

⸻

13. Submission Format

Required columns:

key
fare_amount

Output:

submission.to_csv(..., index=False)

⸻

14. Performance Optimization Rules

Mandatory

* vectorized NumPy
* avoid row-wise apply
* delete unused objects
* use garbage collection
* avoid Python loops

Recommended

* float32 conversion
* categorical encoding
* chunk processing

⸻

15. Medal-Level Improvements

LLM should attempt these for stronger leaderboard performance.

Advanced Geographic Features

Distance to Manhattan center

Borough estimation

Airport route flags

Example:

is_jfk_trip
is_lga_trip

⸻

Target Encoding

For:

* hour
* weekday
* cluster ids

Use out-of-fold encoding only.

⸻

Ensemble Ideas

Blend:

* LightGBM
* XGBoost
* CatBoost

⸻

Cross Validation

Prefer:

* 5-fold CV
* repeated KFold

Store OOF predictions.

⸻

16. Forbidden / Failure Patterns

DO NOT

Use row-wise loops

Bad:

for row in df.iterrows()

Forget radians conversion

Recompute features inconsistently

Train on raw coordinates only

Ignore outliers

Use extremely deep trees

Use massive learning rates

⸻

17. Expected Winning Characteristics

A medal-quality solution should contain:

* robust geographic feature engineering
* airport-aware priors
* temporal decomposition
* distance + bearing features
* strong boosted trees
* aggressive cleaning
* efficient large-scale processing

The main competitive edge is NOT model complexity.

The edge comes from:

1. geographic priors
2. distance engineering
3. noise removal
4. efficient boosting setup