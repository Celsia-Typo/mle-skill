Cdiscount Image Classification Challenge — Gold-Level Solution Blueprint

1. Core Philosophy

This solution is not a simple image classification pipeline.

The dataset has several unique characteristics:

* Products contain multiple images
* Labels are organized hierarchically:
    * Level 1
    * Level 2
    * Level 3 (final category)
* Raw data is stored in massive BSON archives
* Data leakage is extremely easy if product grouping is ignored
* IO throughput becomes a dominant bottleneck

Therefore, the correct strategy is:

Hierarchical Multi-Task Classification
            +
Product-Level Validation Isolation
            +
On-the-Fly BSON Streaming
            +
Distributed GPU Training
            +
Product-Level Logit Aggregation

The key insight is:

The competition is fundamentally PRODUCT classification,
not image classification.

All architectural decisions are built around that principle.

⸻

2. High-Level System Architecture

BSON Dataset
    ↓
Image-Level Metadata Indexing
    ↓
Product-Stratified Split
    ↓
On-the-Fly BSON Decoding
    ↓
EfficientNet-B2 Backbone
    ↓
Hierarchical Multi-Task Heads
    ↓
Image-Level Logits
    ↓
Product-Level Logit Aggregation
    ↓
Final Category Prediction

⸻

3. Dataset Engineering

3.1 BSON Indexing System

The solution never loads BSON files directly during training.

Instead:

Stage 1 — Sequential BSON Scan

Each BSON document is indexed using:

* byte offset
* document length
* product id
* number of images

This converts the raw BSON archive into a random-access database.

Stored metadata:

{
    "_id",
    "offset",
    "length",
    "img_idx"
}

⸻

3.2 Image-Level Expansion

Each product contains multiple images.

The index is expanded into image-level rows:

Product A:
    img0
    img1
    img2
↓
3 independent training samples
sharing the same product ID

This enables:

* mini-batch SGD
* distributed training
* augmentation diversity

while preserving product grouping information.

⸻

3.3 Parallel BSON Parsing

The indexing stage uses multiprocessing.

Why?

BSON decoding becomes CPU-bound.

The solution parallelizes:

Offset Parsing
    +
Metadata Extraction
    +
Image Count Recovery

across 32 workers.

This is critical because:

* train.bson is enormous
* naive parsing becomes prohibitively slow

⸻

4. Leakage Prevention Strategy

4.1 Product-Level Split

This is the single most important component.

Naive image-level splitting causes catastrophic leakage:

Train:
    Product A Image 1
Validation:
    Product A Image 2

The model memorizes the product.

Validation becomes meaningless.

⸻

4.2 Product-Stratified Shuffle Split

The solution splits at the PRODUCT level.

All images belonging to the same product remain inside one split.

Implementation logic:

train_mask = X['_id'].isin(train_pids)

This guarantees:

Product Isolation

across train/validation.

⸻

4.3 Hierarchical Stratification

Stratification is performed using:

Level-1 Category

instead of full 5270-way categories.

Why?

Because:

* full stratification becomes unstable
* many rare classes exist
* level-1 maintains semantic balance

This stabilizes validation statistics.

⸻

5. Preprocessing Pipeline

5.1 Metadata-Only Preprocessing

No images are preloaded into RAM.

Instead, preprocessing only prepares:

offset
length
img_idx
bson_path

This enables:

Streaming Training

without massive memory overhead.

⸻

5.2 Memory Optimization

Critical dtype downcasting:

Field	Type
img_idx	int8
cat_idx	int16
l1_idx	int8
l2_idx	int16

This significantly reduces RAM pressure for tens of millions of rows.

⸻

5.3 Data Integrity Checks

The pipeline validates:

* NaNs
* Infinite values
* alignment consistency
* row correspondence

before training begins.

This prevents silent corruption during large-scale runs.

⸻

6. Dataset Loader Design

6.1 On-the-Fly BSON Decoding

Images are decoded lazily inside __getitem__.

Pipeline:

Seek Offset
    ↓
Read BSON Bytes
    ↓
Decode BSON
    ↓
Extract Image
    ↓
cv2.imdecode
    ↓
Transform

This avoids:

pre-extracted image storage explosion

⸻

6.2 Per-Worker File Handles

Each DataLoader worker maintains its own BSON file handle.

Why?

Shared file handles across workers create:

* IO contention
* race conditions
* corrupted reads

Per-worker handles maximize throughput.

⸻

6.3 OpenCV Decoding

The solution uses:

cv2.imdecode()

instead of PIL loading.

Advantages:

* faster JPEG decoding
* lower overhead
* better multiprocessing compatibility

⸻

7. Model Architecture

7.1 EfficientNet-B2 Backbone

The backbone:

EfficientNet-B2

initialized with ImageNet weights.

Why EfficientNet?

Because Cdiscount requires:

accuracy / FLOPs efficiency

rather than extreme model scale.

EfficientNet provides:

* strong transfer learning
* efficient inference
* stable optimization

⸻

8. Hierarchical Multi-Task Learning

8.1 Three Classification Heads

The model predicts:

Head	Task
Head 1	Level-1
Head 2	Level-2
Head 3	Final Category

Architecture:

Backbone Features
    ├── L1 Head
    ├── L2 Head
    └── L3 Head

⸻

8.2 Why Multi-Task Works

The hierarchy provides semantic regularization.

Example:

Electronics
    → Phones
        → iPhone Accessories

Predicting higher-level categories helps:

* stabilize features
* improve representation learning
* reduce overfitting
* accelerate convergence

⸻

8.3 Loss Weighting

Loss formulation:

0.1 * L1
+ 0.1 * L2
+ 0.8 * L3

The final category dominates training,
while hierarchy acts as auxiliary supervision.

⸻

9. Training System

9.1 Distributed Data Parallel (DDP)

Training uses:

PyTorch DDP

across all GPUs.

Why DDP?

The dataset is extremely large.

Single-GPU training becomes too slow.

⸻

9.2 Mixed Precision Training

The solution uses:

torch.amp.autocast()
GradScaler

Benefits:

* lower VRAM usage
* larger batch sizes
* faster throughput

without significant accuracy loss.

⸻

9.3 OneCycleLR

Learning rate scheduling:

OneCycleLR

Advantages:

* faster convergence
* better generalization
* stable large-batch training

⸻

9.4 Label Smoothing

Loss:

CrossEntropyLoss(label_smoothing=0.1)

This reduces:

* overconfidence
* noisy-label sensitivity
* class imbalance instability

⸻

10. Augmentation Strategy

The baseline augmentation is intentionally minimal:

RandomHorizontalFlip()

Why?

Cdiscount product images are relatively standardized.

Heavy augmentation risks:

* destroying fine-grained product details
* altering product semantics

This solution prioritizes:

feature preservation

over aggressive regularization.

⸻

11. Validation & Inference

11.1 Image-Level Prediction

Each image produces:

5270-dimensional logits

These are NOT final predictions.

⸻

11.2 Product-Level Aggregation

This is the true core of the solution.

For each product:

Multiple Image Logits
        ↓
Mean Aggregation
        ↓
Product Logits
        ↓
Argmax

Why averaging logits works:

* reduces noisy image influence
* stabilizes prediction confidence
* integrates multi-view information

⸻

11.3 Why Logit Averaging Beats Probability Averaging

Averaging logits preserves:

relative confidence geometry

before softmax distortion.

This usually improves ensemble calibration.

⸻

12. DDP Prediction Reconstruction

Distributed inference breaks global ordering.

The solution fixes this using:

(global_index, logits)

pairs.

Final reconstruction:

final_logits[indices] = logits

This guarantees:

exact row alignment restoration

after multi-GPU inference.

⸻

13. Ensemble Philosophy

Although only one model is shown,
the framework supports:

multi-backbone ensembling

through:

all_val_preds
all_test_preds

The intended extension is:

Model	Role
EfficientNet-B2	strong baseline
ConvNeXt	texture modeling
ViT	global semantics
NFNet	robust optimization

⸻

14. Scalability Design

The entire pipeline is designed around scalability.

Key engineering decisions:

Problem	Solution
Massive BSON size	Random-access indexing
IO bottleneck	Streaming decode
RAM explosion	Metadata-only preprocessing
GPU utilization	DDP
Product leakage	Product-level splitting
Multi-image ambiguity	Logit aggregation

⸻

15. Why This Solution Gets Medals

This is not merely a better CNN.

It solves the competition at the SYSTEM level.

Most competitors fail because they:

* split at image level
* ignore product grouping
* preload images inefficiently
* underutilize GPUs
* ignore hierarchical structure

This solution correctly models:

Product-Level Multi-View Hierarchical Classification

which aligns with the actual leaderboard objective.

⸻

16. Expected Performance Characteristics

Strengths

* Excellent scalability
* Strong validation reliability
* High GPU utilization
* Robust product-level inference
* Strong generalization

Weaknesses

* BSON decoding still expensive
* EfficientNet-B2 not state-of-the-art today
* Minimal augmentations
* Single-fold validation variance possible

⸻

17. Gold-Level Upgrade Directions

17.1 Stronger Backbones

Upgrade to:

* ConvNeXt
* EVA
* ViT
* Swin Transformer

⸻

17.2 Better Aggregation

Replace mean logits with:

Attention Pooling
GeM Pooling
Learned Product Aggregation

⸻

17.3 Advanced Augmentations

Possible additions:

* RandAugment
* Mixup
* CutMix
* Random Erasing

⸻

17.4 Hierarchical Label Constraints

Enforce valid taxonomy transitions:

L1 → L2 → L3

during inference.

⸻

17.5 Multi-Fold OOF Ensembling

True gold-tier systems typically use:

5-fold product-level CV

with OOF blending.

⸻

18. Final Blueprint Summary

The complete winning recipe is:

BSON Streaming System
    +
Product-Level Leakage Prevention
    +
Hierarchical Multi-Task Learning
    +
EfficientNet Backbone
    +
Distributed Mixed Precision Training
    +
Product-Level Logit Aggregation

The most important ideas are:

1. Treat products — not images — as the learning unit
2. Prevent leakage at the product level
3. Aggregate predictions across product images
4. Use hierarchy-aware supervision
5. Optimize IO as aggressively as model training

That combination is what transforms a normal classifier
into a medal-level Cdiscount solution.

⸻

19. Anti-Blueprint — Execution Failure Patterns To Avoid

Core Principle

In large-scale Kaggle training pipelines, the primary failure mode is often not model quality, but incomplete execution.

A run is INVALID if any of the following occurs:

* Python traceback
* Timeout
* Missing final metric
* Missing submission
* Checkpoint reload failure
* Schema mismatch crash
* Incomplete validation
* Interrupted inference pipeline

Even strong intermediate metrics are irrelevant if the pipeline does not complete end-to-end.

Therefore:

Execution robustness is a first-class optimization target.

⸻

19.1 Checkpoint Serialization Failure (PyTorch 2.6)

Failure Pattern

Training completes partially, but checkpoint reload crashes:

checkpoint = torch.load(best_path, map_location=device)

PyTorch 2.6 defaults to:

weights_only=True

If the checkpoint contains:

* numpy arrays
* custom classes
* unsupported objects

the safe unpickler throws:

_pickle.UnpicklingError:
Weights only load failed

This bypasses:

* best checkpoint reload
* final validation
* test inference
* submission generation

Result:

* no final metric
* invalid run

⸻

Root Cause

Unsafe checkpoint contents:

{
    'model': state_dict,
    'idx_to_category': numpy_array
}

NumPy arrays are not allowlisted by weights-only loading.

⸻

Prevention Blueprint

Option A (Recommended)

Explicitly disable safe weights-only loading:

checkpoint = torch.load(
    best_path,
    map_location=device,
    weights_only=False
)

ONLY if checkpoint is locally trusted.

⸻

Option B (Production-Safe)

Save only:

* tensors
* primitive Python types

Example:

torch.save({
    'model': model.state_dict(),
    'idx_to_category':
        torch.as_tensor(idx_to_category),
    'score': float(best_score),
    'epoch': int(epoch),
}, best_path)

Reload:

idx_to_category = (
    checkpoint['idx_to_category']
    .cpu()
    .numpy()
)

⸻

Option C

Store metadata separately:

np.save(...)

and keep PyTorch checkpoint model-only.

⸻

19.2 Runtime Timeout Failure

Failure Pattern

Training begins correctly.

Intermediate metrics appear:

Val Metric: 0.66

but execution exceeds Kaggle runtime limit:

TimeoutError:
Execution exceeded the time limit

or:

KeyboardInterrupt

before:

* final metric
* checkpoint reload
* submission generation

⸻

Root Cause

Pipeline cost exceeds wall-clock budget.

Typical causes:

* repeated BSON decoding
* full validation every epoch
* huge image-level datasets
* expensive product-level aggregation
* excessive epochs
* oversized models
* no runtime awareness

⸻

Prevention Blueprint

1. Time-Aware Training

Track wall-clock time:

start_time = time.time()

After each epoch:

elapsed = time.time() - start_time
remaining = LIMIT - elapsed

Early-stop BEFORE timeout.

⸻

2. Reserve Inference Budget

Always reserve time for:

* checkpoint reload
* validation
* test inference
* submission writing

Never consume full runtime in training.

⸻

3. Validation Budgeting

Avoid:

full validation every epoch

Instead:

* validate every N epochs
* use validation subsets
* cache embeddings/logits
* perform full validation once

⸻

4. Reduce I/O Bottlenecks

Avoid repeated BSON decompression.

Use:

* LMDB
* WebDataset
* extracted shards
* cached metadata
* memory-mapped indices

⸻

5. Runtime Guards

Mandatory:

try:
    ...
finally:
    print(f"Final Validation Score: {best_score}")

Even partial runs should emit recoverable metrics.

⸻

6. Resume-Safe Training

Periodic checkpoints:

save_every_n_steps

Store:

* epoch
* optimizer
* scheduler
* scaler
* RNG state

⸻

19.3 Dataset Schema Fragility

Failure Pattern

Code assumes exact column names:

level1
level_1
l1

Dataset actually contains:

category_level1
category_level2

Result:

RuntimeError:
category_names.csv must contain
level1 and level2 hierarchy columns

Training never starts.

⸻

Root Cause

Overly strict schema assumptions.

⸻

Prevention Blueprint

Flexible Column Matching

Use normalized matching:

normalized = (
    c.lower().replace('_', '')
)

Example:

level1_col = next(
    (
        c for c in cols
        if normalized(c)
        in ('level1', 'categorylevel1')
    ),
    None
)

⸻

Diagnostic Logging

Before raising:

print(category_df.columns.tolist())

to expose schema mismatches immediately.

⸻

Schema Validation Stage

Add explicit preflight validation BEFORE training:

validate_schema()
validate_paths()
validate_labels()

Never discover dataset issues after training starts.

⸻

19.4 Function Signature Drift

Failure Pattern

Function definition:

def evaluate(model, loader, device):

Call site:

evaluate(..., scheduler=scheduler)

Result:

TypeError:
unexpected keyword argument

Validation crashes after training.

⸻

Root Cause

Training loop evolved independently from helper function API.

Common in iterative Kaggle development.

⸻

Prevention Blueprint

1. Stable Interface Layer

Centralize pipeline APIs.

Avoid ad-hoc helper evolution.

⸻

2. Dry-Run Validation

Before long training:

Run:

1 batch train
1 batch val
1 batch inference

to verify:

* signatures
* tensor shapes
* AMP
* scheduler
* submission path

⸻

3. End-to-End Smoke Test

Mandatory before full run:

DEBUG_MODE = True

with:

* tiny subset
* 1 epoch
* full submission pipeline

Goal:

verify COMPLETE execution.

Not accuracy.

⸻

19.5 Missing Final Metric

Failure Pattern

Intermediate logs exist:

Epoch 1 Val Metric: 0.66

But evaluator expects:

Final Validation Score:

Missing final print = invalid run.

⸻

Prevention Blueprint

Always emit:

print(
    f"Final Validation Score: "
    f"{best_score:.6f}"
)

AFTER:

* checkpoint reload
* final validation
* BEFORE inference

and preferably inside:

finally:

⸻

19.6 Submission Integrity Failure

Failure Pattern

Inference partially runs but submission integrity is unchecked.

Possible issues:

* wrong row count
* missing IDs
* NaNs
* duplicate predictions

⸻

Prevention Blueprint

Mandatory assertions:

assert len(submission) == TOTAL_ROWS
assert submission.isnull().sum().sum() == 0
assert submission['category_id'].dtype == int

Always save:

submission.csv

before program exit.

⸻

19.7 Blueprint Drift

Failure Pattern

Code comments claim:

3-stage XCeption protocol

Actual implementation:

* EfficientNet-B2
* single-stage training
* OneCycle
* no checkpoint chaining

This creates:

* protocol inconsistency
* unverifiable reproduction
* architecture drift

⸻

Prevention Blueprint

Blueprints must define:

* architecture
* stage schedule
* optimizer
* LR policy
* checkpoint chain
* augmentation
* runtime budget

Implementation must match blueprint exactly.

⸻

Final Engineering Principle

For large Kaggle systems:

A slower fully-completing pipeline
beats
a stronger pipeline that crashes.

The true optimization target is:

Expected leaderboard score
under runtime and execution constraints

NOT:

peak validation metric
