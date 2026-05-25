GOLD-STABLE BLUEPRINT — CHAMPS Scalar Coupling (Multi-Task Neural Network Medal Solution)

⸻

1. CORE PHILOSOPHY

This competition is NOT solved by:

* blindly minimizing MAE
* training one global neural network
* random feature engineering
* leaderboard overfitting
* ignoring coupling-type distribution
* naive validation
* optimizing auxiliary targets incorrectly
* leaking test information through preprocessing
* averaging CV MAE directly

This competition IS solved by:

* type-specific modeling
* stable log-MAE optimization
* physically meaningful geometric features
* auxiliary quantum targets
* molecule-aware validation
* carefully normalized multi-task learning
* strict metric alignment with Kaggle
* robust numerical preprocessing
* preventing hidden catastrophic type failures

The biggest failure mode in this competition is:

CV metric appears strong while Kaggle score becomes positive or catastrophically bad.

This almost always comes from:

* incorrect metric aggregation
* unstable per-type predictions
* one coupling type collapsing
* leakage
* auxiliary loss domination
* wrong validation split
* prediction scale instability
* NaN/inf propagation
* inconsistent feature scaling

Your entire pipeline must be engineered to prevent this.

⸻

2. COMPETITION METRIC — CRITICAL

Official Metric

The official metric is:

\frac{1}{T} \sum_{t=1}^{T} \log \left( \frac{1}{n_t} \sum_{i=1}^{n_t} |y_i - \hat{y}_i| \right)

NOT:

* global MAE
* RMSE
* weighted MAE
* averaged fold MAE

The metric is:

* computed separately per coupling type
* THEN log-transformed
* THEN averaged

⸻

3. ABSOLUTE METRIC SAFETY RULES

NEVER compute validation like this

mae = mean_absolute_error(y_true, y_pred)

OR

np.mean(np.abs(y_true - y_pred))

across all rows globally.

This produces fake-good CV.

⸻

ALWAYS compute:

scores = []
for t in coupling_types:
    mask = (types == t)
    mae_t = np.mean(np.abs(
        y_true[mask] - y_pred[mask]
    ))
    mae_t = max(mae_t, 1e-9)
    scores.append(np.log(mae_t))
final_score = np.mean(scores)

⸻

4. VALIDATION STRATEGY — NON-NEGOTIABLE

NEVER use random row split

Random splitting leaks molecule structure.

This creates:

* fake CV
* positive LB scores
* severe generalization collapse

⸻

REQUIRED

Use:

GroupKFold

with:

groups = molecule_name

Every molecule must exist in only ONE fold.

⸻

5. TYPE-SPECIFIC MODELING

REQUIRED

Train separate models per coupling type.

Examples:

* 1JHC
* 2JHH
* 3JHN
* etc.

Reason:

Each type has:

* different scale
* different distribution
* different geometry
* different physics

Single global NN underfits.

⸻

6. MULTI-TASK LEARNING PHILOSOPHY

The scalar coupling target alone is noisy.

Auxiliary quantum targets stabilize learning.

Use auxiliary outputs:

* Mulliken charges
* shielding tensors
* tensor off-diagonal terms

These act as:

* physical regularizers
* latent structure supervision
* geometric attractors

⸻

7. AUXILIARY LOSS SAFETY

CRITICAL

Auxiliary targets MUST NOT dominate scalar coupling training.

This is one of the main causes of:

* excellent auxiliary MAE
* terrible competition score

⸻

REQUIRED

Scale auxiliary targets carefully.

Example:

m1 = 1
m2 = 4
m3 = 1

Then:

aux = scaler.fit_transform(aux)
aux *= multiplier

⸻

REQUIRED

Main target must dominate optimization.

Recommended loss weights:

loss_weights = {
    "scalar": 1.0,
    "charge": 0.05,
    "tensor_diag": 0.02,
    "tensor_offdiag": 0.02,
}

NEVER allow auxiliary losses to exceed main loss magnitude.

⸻

8. FEATURE ENGINEERING — MEDAL CORE

The NN solution depends heavily on geometry features.

REQUIRED FEATURE CATEGORIES

Pair Geometry

* dx
* dy
* dz
* Euclidean distance

⸻

Molecule Center Features

Compute molecule centroid:

c_x
c_y
c_z

Then:

* distance to center
* normalized center vectors

⸻

Closest/Farthest Atom Features

For each atom:

* nearest atom
* farthest atom

Generate:

* distances
* vectors
* cosine similarities

⸻

Angular Features

Critical:

cos(a,b)

between:

* bond vectors
* center vectors
* nearest-neighbor vectors
* farthest-neighbor vectors

These features are extremely important.

⸻

9. FEATURE ENGINEERING SAFETY RULES

NEVER divide without epsilon

ALWAYS:

v / (norm + 1e-10)

Otherwise:

* NaNs
* exploding validation
* positive LB

⸻

ALWAYS verify:

np.isfinite(features).all()

before training.

⸻

10. DATA LEAKAGE RULES

NEVER fit scalers on:

train + validation

inside folds.

This silently leaks.

⸻

STRICT RULE

Inside each fold:

scaler.fit(train_only)
scaler.transform(valid)
scaler.transform(test)

⸻

11. NORMALIZATION

REQUIRED

Use:

StandardScaler

for dense NN features.

Avoid:

* MinMaxScaler
* RobustScaler

unless experimentally validated.

⸻

12. NETWORK ARCHITECTURE

Medal-stable architecture characteristics

* deep dense network
* wide hidden layers
* BatchNorm everywhere
* LeakyReLU preferred
* moderate dropout
* multi-head outputs

⸻

REQUIRED DESIGN

Shared trunk:

Input
→ Dense
→ BN
→ LeakyReLU
→ Dropout
(repeated)

Outputs:

* scalar coupling
* charges
* tensor diagonal
* tensor off-diagonal

⸻

13. ACTIVATION RULES

Preferred:

LeakyReLU(alpha=0.05)

Avoid plain ReLU dead neurons.

⸻

14. REGULARIZATION

REQUIRED

BatchNorm after almost every Dense layer.

Dropout:

0.1–0.4

Too much dropout causes underfitting.

Too little causes fold instability.

⸻

15. OPTIMIZER

Preferred:

Adam

starting LR:

1e-3

or

3e-4

⸻

16. LEARNING RATE STRATEGY

REQUIRED

Use:

ReduceLROnPlateau

Example:

factor=0.1
patience=7
min_lr=1e-6

This competition strongly benefits from long convergence tails.

⸻

17. EARLY STOPPING

REQUIRED

restore_best_weights=True

Without this:

* later epochs often degrade LB
* hidden type collapse occurs

⸻

18. MEMORY ENGINEERING

This competition is memory sensitive.

REQUIRED

Downcast aggressively:

float64 → float32/float16
int64 → int32/int16

⸻

REQUIRED

Merge carefully:

drop_duplicates()

before joins.

Otherwise test rows explode.

⸻

19. MERGE SAFETY

After every merge:

VERIFY

assert len(df_after) == len(df_before)

This catches catastrophic duplication bugs.

⸻

20. PREDICTION STABILITY

REQUIRED

Before submission:

assert np.isfinite(preds).all()

and

assert not np.isnan(preds).any()

⸻

REQUIRED

Per type:

print(
    t,
    preds.min(),
    preds.max(),
    preds.mean(),
    preds.std()
)

Collapsed predictions on one type can destroy LB.

⸻

21. CV MONITORING — MANDATORY

Always print:

per_type_log_mae

NOT just total score.

One bad type can produce positive LB even if others look excellent.

⸻

22. COMMON FAILURE MODES

FAILURE #1

Using global MAE.

Result:

* fake CV
* positive LB

⸻

FAILURE #2

Random split.

Result:

* leakage
* leaderboard collapse

⸻

FAILURE #3

Auxiliary loss domination.

Result:

* physics targets improve
* competition metric worsens

⸻

FAILURE #4

NaN cosine features.

Result:

* unstable predictions
* catastrophic folds

⸻

FAILURE #5

One coupling type collapse.

Result:

* positive overall score

Even if 7/8 types are good.

⸻

23. ENSEMBLE STRATEGY

Best medal setups usually combine:

* LightGBM
* Neural Networks

NN captures:

* geometric smoothness
* latent quantum structure

LGBM captures:

* local tabular interactions

⸻

24. NN ENSEMBLE RULES

Strong ensemble diversity:

* different seeds
* different dropout
* different widths
* different auxiliary weights
* different fold splits

Average predictions.

⸻

25. FINAL SUBMISSION SAFETY CHECKLIST

Before generating submission:

REQUIRED CHECKS

Shape

len(test_preds) == len(sample_submission)

⸻

Finite values

np.isfinite(test_preds).all()

⸻

Per-type sanity

Verify:

* no constant predictions
* no exploding std
* no absurd ranges

⸻

Fold consistency

Per-fold scores should be stable.

Large variance means leakage or collapse.

⸻

26. MEDAL-STABLE TRAINING LOOP

REQUIRED ORDER

1. Load raw data
2. Reduce memory
3. Merge structures safely
4. Create geometry features
5. Create cosine features
6. Validate no NaNs
7. Split by molecule groups
8. Scale train-only
9. Train per coupling type
10. Monitor per-type log-MAE
11. Save best checkpoints
12. Predict test
13. Aggregate carefully
14. Validate submission integrity

⸻

27. GOLD-LEVEL EXTENSIONS

Potential improvements:

* residual dense blocks
* gated MLP
* type embeddings
* molecule graph embeddings
* snapshot ensembling
* cyclic LR
* rank-gauss target transforms
* adversarial validation
* pseudo-labeling
* fold-specific auxiliary weights

⸻

28. ABSOLUTE NON-NEGOTIABLE RULES

NEVER

* random split molecules
* compute global MAE
* ignore per-type scores
* allow NaNs
* fit scaler outside folds
* merge without row-count validation
* trust leaderboard over CV
* use auxiliary losses without weighting
* submit without per-type sanity checks

Failure on ANY of these commonly produces:

positive leaderboard scores despite apparently good local metrics.