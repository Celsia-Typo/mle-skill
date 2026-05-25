GOLD-STABLE BLUEPRINT — Statoil Iceberg Classifier (2nd-Place Style)

1. CORE PHILOSOPHY

This competition is NOT solved by:

* one giant CNN
* handcrafted radar heuristics
* angle leakage into CNN
* heavy postprocessing
* maximizing single-model strength

The competition is solved by:

Stable Multi-Stage Ensemble Refinement

through:

* many medium-strength CNNs
* carefully controlled CV
* incidence-angle statistical exploitation
* train+test joint grouping
* lightweight meta-learning
* ultra-conservative boosting

The essential insight:

incidence angle contains strong hidden distribution structure,
but directly feeding it into CNNs causes overfitting.

Therefore:

* CNNs learn ONLY image features
* angle information is exploited later
* stacking model combines both safely

⸻

2. HIGH-LEVEL PIPELINE

Raw SAR bands
    ↓
Train many CNNs with CV
    ↓
Generate OOF predictions
    ↓
Find best NN subset
    ↓
Create angle-group statistics
    ↓
Create KNN angle feature
    ↓
Train tiny LightGBM stacker
    ↓
Final clipped probabilities

⸻

3. DATA REPRESENTATION

3.1 Input Construction

Use ONLY:

* band_1
* band_2

Ignore:

* inc_angle during CNN training

Construct image tensor:

X = np.stack([band_1, band_2], axis=-1)
shape = (75, 75, 2)

NO:

* handcrafted channels
* FFT transforms
* angle concatenation
* metadata fusion inside CNN

The CNN must specialize purely on visual structure.

⸻

4. CNN STAGE

⸻

4.1 CNN Architecture Philosophy

Use:

* shallow-to-medium CNN
* VGG-style blocks
* aggressive dropout
* small parameter count
* stable optimization

Avoid:

* ResNet depth explosion
* extremely large models
* attention modules
* overengineered architectures

The competition dataset is too small for giant models.

⸻

4.2 Canonical Architecture

Structure

Input(75x75x2)
BN
Conv(32,7x7)
MaxPool
Dropout
Conv(64,5x5)
MaxPool
Dropout
Conv(128,3x3)
MaxPool
Dropout
Flatten
Dense(128)
Dropout
Sigmoid

Critical Details

BatchNorm

Use:

momentum=0.0

This is unusually important for stability.

Optimizer

Use:

Adam(lr=1e-4)

Loss

binary_crossentropy

⸻

4.3 Data Augmentation

This is one of the MOST IMPORTANT parts.

Use ONLY:

rotation_range = 10
shift_range = 0.12
vertical_flip = True

Critical discovery:

0.12 * 75 ≈ 9 pixels

This shift amount aligns well with SAR object displacement.

Avoid:

* huge rotations
* elastic transforms
* color jitter
* horizontal flip abuse

SAR imagery behaves differently from natural images.

⸻

4.4 Cross Validation

Use:

StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

Critical:

ALL ensemble models MUST use:

* same folds
* same split logic

to ensure aligned OOF predictions.

⸻

4.5 Multi-NN Diversity

Train MANY independent CNN runs:

5~20 models

Diversity sources:

* random seed
* initialization
* augmentation randomness

NOT:

* wildly different architectures

The winning strategy prefers:

stable weak diversity

over:

chaotic heterogeneous ensembles

⸻

4.6 OOF Prediction Generation

For every fold:

Validation

Generate:

OOF[val_idx]

Test

Generate:

test_preds_fold

Final test prediction:

mean(test_preds_fold)

Store:

train_preds_SEED.npy
test_preds_SEED.npy

These become stacker features.

⸻

4.7 Test-Time Augmentation (TTA)

Use ONLY:

original
vertical_flip

Prediction:

pred = (pred_orig + pred_flip) / 2

Do NOT use:

* 8-way TTA
* rotation TTA
* heavy geometric TTA

Small conservative TTA works best.

⸻

5. BEST SUBSET ENSEMBLING

This is a crucial hidden trick.

⸻

5.1 Brute Force Subset Search

Given multiple CNN predictions:

train_nn_preds.shape = [N_models, N_samples]

Search ALL subsets:

for subset in powerset(models):
    score = log_loss(...)

Select:

subset with best CV loss

This is feasible because:

#models < 30

⸻

5.2 Final Ensemble Prediction

Use:

mean(best_subset_predictions)

NOT:

* weighted averaging
* greedy hill climbing
* stacking at this stage

Simple mean is more stable.

⸻

6. ANGLE-BASED STATISTICAL FEATURES

This is the TRUE winning insight.

⸻

6.1 Key Discovery

Objects with similar:

inc_angle

often share similar labels.

Instead of feeding angle to CNN:

exploit it statistically later.

⸻

6.2 Joint Train+Test Grouping

CRITICAL:

Combine:

train + test

before grouping.

Otherwise:

* train distribution differs
* leakage-like overfitting occurs

⸻

6.3 Group Features

Group by:

inc_angle

For each group compute:

median prediction
mean prediction
group size

Features:

med
mea
cnt

⸻

6.4 Leakage-Safe Refinement Trick

For training samples ONLY:

temporarily replace:

prediction ↔ ground truth

during aggregation.

This boosts statistical precision while avoiding direct self-leakage.

Pseudo:

best[idx], pred[idx] = pred[idx], best[idx]
compute stats
swap back

This subtle trick is extremely important.

⸻

7. KNN ANGLE FEATURE

For rare/unique angles:

group statistics fail.

Therefore:

train KNN regressor:

KNeighborsRegressor(
    n_neighbors=23,
    weights='distance',
    algorithm='brute'
)

Input:

inc_angle

Target:

nn_pred

Output feature:

knn_pred

⸻

7.1 Important Constraints

Exclude:

* fake angles
* current sample itself

during fitting.

Fake angle detection:

len(decimal_part) > 5

These correspond to suspicious synthetic angles.

⸻

8. LIGHTGBM STACKER

⸻

8.1 Feature Set

ONLY 5 FEATURES:

[
    nn_pred,
    med,
    mea,
    cnt,
    knn_pred
]

The stacker is intentionally tiny.

⸻

8.2 LightGBM Philosophy

Use:

* shallow trees
* tiny model
* conservative boosting

Avoid:

* deep trees
* huge estimators
* aggressive fitting

⸻

8.3 Canonical Parameters

LGBMClassifier(
    max_depth=3,
    n_estimators=70,
    learning_rate=0.1,
    min_child_samples=40
)

⸻

8.4 Massive CV

Use:

StratifiedKFold(
    n_splits=51,
    shuffle=True,
    random_state=27
)

This produces:

* robust validation
* smooth probability estimates

⸻

9. FINAL PROBABILITY CLIPPING

Always clip:

np.clip(preds, 0.001, 0.999)

Purpose:

* avoid logloss explosions
* stabilize leaderboard variance

This is mostly precautionary.

⸻

10. CRITICAL DESIGN RULES

⸻

RULE 1 — NEVER FEED ANGLE TO CNN

This causes:

* shortcut learning
* severe overfitting
* weak generalization

Angle must ONLY appear in meta-stage.

⸻

RULE 2 — OOF PREDICTIONS ARE SACRED

Every stacking feature MUST come from:

strict out-of-fold predictions

Never leak full-train predictions.

⸻

RULE 3 — TRAIN+TEST JOINT STATISTICS

Angle grouping MUST include:

both train and test

Otherwise:

meta-features become distribution-biased.

⸻

RULE 4 — SMALL MODELS WIN

The dataset is tiny.

Prefer:

stable small CNNs

over:

massive architectures

⸻

RULE 5 — CONSERVATIVE ENSEMBLING

The solution wins through:

many small reliable gains

NOT:

one giant breakthrough model

⸻

11. EXPECTED PERFORMANCE CHARACTERISTICS

Typical behavior:

Stage	Effect
Single CNN	decent
CNN ensemble	large gain
Angle stats	surprisingly huge gain
KNN feature	small stable gain
LightGBM	final refinement

The angle-statistics stage contributes disproportionately.

⸻

12. IMPLEMENTATION BLUEPRINT FOR LLM CODE GENERATION

An LLM implementing this solution should generate:

Module 1 — nn.py

Responsibilities:

* load SAR bands
* build VGG-style CNN
* augmentation
* CV training
* TTA inference
* save OOF/test predictions

Outputs:

train_preds_*.npy
test_preds_*.npy

⸻

Module 2 — features.py

Responsibilities:

* load all NN predictions
* brute-force best subset search
* generate angle-group statistics
* generate KNN angle feature
* save feature table

Outputs:

features.csv

⸻

Module 3 — lgbm.py

Responsibilities:

* load features
* LightGBM CV
* final LightGBM training
* generate clipped submission

Outputs:

preds.csv

⸻

13. FORBIDDEN PATTERNS

The generated implementation MUST NOT:

CNN Stage

* feed inc_angle into CNN
* use pretrained ImageNet models
* use RGB assumptions
* use huge architectures
* use segmentation-style pipelines

Ensemble Stage

* use train predictions instead of OOF
* compute statistics separately for train/test
* use leakage-prone aggregation

LightGBM Stage

* use deep trees
* use huge feature engineering
* use aggressive hyperparameter search

⸻

14. META-INSIGHT

This solution is fundamentally:

distribution exploitation

rather than:

pure visual recognition

The CNN extracts:

object appearance priors

The meta-model extracts:

hidden incidence-angle structure

The winning score comes from combining both carefully while avoiding leakage.