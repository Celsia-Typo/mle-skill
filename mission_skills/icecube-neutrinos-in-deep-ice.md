IceCube 10th Place Solution Reproduction Blueprint (LLM Code Generation Guide)

Project Goal

Reproduce and extend the
IceCube - Neutrinos in Deep Ice
10th Place Solution training and inference pipeline.

This blueprint is designed to guide an LLM to automatically generate:

* data preprocessing code
* Dataset/DataLoader
* PyTorch Lightning training framework
* four model architectures
* custom loss functions
* inference pipeline
* ensemble system
* validation strategy
* submission generation
* project structure

⸻

1. Core Objectives

The generated code MUST include:

Required Components

Data

* parquet/batch loading
* variable-length pulse sequence support
* pulse truncation
* sequence padding
* packed sequence processing

Models

Implement all four architectures:

* GRU classification model (Model1)
* BiLSTM regression model (Model2)
* Embedding + BiLSTM model (Model3)
* Small Embedding + BiLSTM model (Model4)

Training

* PyTorch Lightning
* mixed precision training
* cosine scheduler
* warmup scheduler
* gradient clipping
* gradient accumulation

Losses

Implement:

* CrossEntropyLoss
* VonMisesFisher3DLoss

Inference

Implement:

* efficient batch inference
* multi-model ensemble
* weighted blending
* normalized directional prediction

⸻

2. Required Project Structure

The LLM MUST generate the following structure:

icecube_solution/
│
├── configs/
│   ├── model1.yaml
│   ├── model2.yaml
│   ├── model3.yaml
│   ├── model4.yaml
│
├── models/
│   ├── model1.py
│   ├── model2.py
│   ├── model3.py
│   ├── model4.py
│   ├── losses.py
│
├── data/
│   ├── dataset.py
│   ├── preprocess.py
│   ├── collate.py
│
├── training/
│   ├── lightning_module.py
│   ├── train.py
│   ├── optimizer.py
│   ├── scheduler.py
│
├── inference/
│   ├── infer.py
│   ├── ensemble.py
│
├── utils/
│   ├── metrics.py
│   ├── geometry.py
│   ├── seed.py
│
├── notebooks/
│
└── README.md

⸻

3. Input Features

Each event/pulse MUST contain exactly 9 features:

features = [
    "sensor_x",
    "sensor_y",
    "sensor_z",
    "time",
    "charge",
    "auxiliary",
    "is_main_sensor",
    "is_deep_veto",
    "is_deep_core",
]

Input tensor shape:

[B, T, 9]

Where:

* B = batch size
* T = variable pulse sequence length

⸻

4. Preprocessing Rules

Model1 Preprocessing

MUST implement:

sensor_x /= 600
sensor_y /= 600
sensor_z /= 600
time = time / 1000
time = time - time.min()
charge /= 300

⸻

Models2-4 Preprocessing

MUST implement:

sensor_x /= 500
sensor_y /= 500
sensor_z /= 500
time = (time - 1.0e4) / 3.0e4
charge = log10(charge + 1)
charge /= 3.0

⸻

5. Dataset Blueprint

The LLM MUST implement:

IceCubeDataset

Features:

* parquet loading
* event grouping
* pulse truncation
* padding-ready outputs
* label extraction

⸻

Dataset Output Format

{
    "x": tensor[T, 9],
    "length": int,
    "target": tensor[3]
}

⸻

Pulse Truncation

MUST define:

MAX_PULSES = 128

If pulse count exceeds max:

* random sampling
    OR
* time-based truncation

⸻

6. Collate Function

The LLM MUST implement:

pad_sequence
pack_padded_sequence

Outputs:

x_padded
lengths
targets

lengths MUST remain CPU tensors.

⸻

7. Model1 Blueprint (GRU Classification)

Architecture

GRU(
    input_size=9,
    hidden_size=192,
    num_layers=3,
    bidirectional=True
)

⸻

Pooling

MUST use:

x.sum(dim=1) / lengths

FORBIDDEN:

* max pooling
* attention pooling

⸻

Head

Linear(384,256)
ReLU
Linear(256, 31*31)

⸻

Output Shape

[B, 961]

⸻

Loss

MUST use:

CrossEntropyLoss

⸻

Labels

The LLM MUST implement:

* direction-to-bin conversion
* spherical discretization
* azimuth/zenith binning

Bin count:

31 x 31

⸻

8. Models2-4 Blueprint (Directional Regression)

Output Format

Models MUST output:

[x, y, z, kappa]

Where:

* xyz = normalized direction vector
* kappa = concentration parameter

⸻

9. VonMisesFisher3DLoss

The LLM MUST implement:

loss = -kappa * cosine_similarity + log_normalizer

Requirements:

* numerical stability
* epsilon protection
* normalized vectors

FORBIDDEN:

* plain MSE loss

⸻

10. Model2 Blueprint

Architecture

BiLSTM(
    input_size=9,
    hidden_size=256,
    num_layers=3,
    dropout=0.2,
    bidirectional=True
)

⸻

Head

Linear(512,256)
ReLU
Dropout(0.2)
Linear(256,3)

⸻

Output Normalization

MUST implement:

kappa = norm(pred)
pred_xyz = pred / kappa

Final output:

[pred_x, pred_y, pred_z, kappa]

⸻

11. Model3 Blueprint

Embedding

Linear(9,512)

Then:

BiLSTM(512 -> 256)

⸻

12. Model4 Blueprint

Embedding

Linear(9,192)

Then:

BiLSTM(192 -> 96)

⸻

13. LightningModule Requirements

The generated LightningModule MUST implement:

training_step

Including:

* forward pass
* loss computation
* metric logging

⸻

validation_step

Including:

* angular error
* validation score tracking

⸻

configure_optimizers

MUST use:

Adam
CosineAnnealingLR

Must support:

* warmup
* min_lr

⸻

14. Training Hyperparameters

The generated code MUST support:

batch_size = 2048
max_lr = 1e-3 or 5e-4
min_lr = 1e-6
epochs = 10~15
warmup_steps = 2000

⸻

15. Training Features

The LLM MUST generate:

AMP Training

precision=16

⸻

Gradient Clipping

gradient_clip_val

⸻

Deterministic Seeding

MUST seed:

torch
numpy
random

⸻

16. Validation Split

MUST use:

train: batches 11-660
valid: batches 1-10

FORBIDDEN:

* random KFold
* random validation split

⸻

17. Metric

The LLM MUST implement:

Angular Error

angle = arccos(
    dot(pred, target)
)

Units:

radians

⸻

18. Inference Blueprint

Inference Flow

MUST implement:

1. checkpoint loading
2. batch inference
3. prediction collection
4. vector normalization
5. ensemble blending

⸻

Ensemble

Support:

* weighted average
* hill climbing blend
* Nelder-Mead optimization

⸻

19. Submission Generation

The generated submission MUST contain:

event_id
azimuth
zenith

⸻

20. Forbidden Rules

The LLM MUST NOT generate:

Data

❌ loading all pulses into RAM
❌ batches without padding
❌ fixed-length sequence assumptions

⸻

Models

❌ Transformer-only replacements
❌ GNN-only replacements
❌ CNN-only replacements

⸻

Losses

❌ replacing VMF loss with MSE
❌ BCE loss

⸻

Pooling

❌ attention pooling
❌ CLS-token pooling

⸻

Training

❌ random train/valid split
❌ shuffled validation batches

⸻

Inference

❌ unnormalized directional vectors
❌ single-model final submission

⸻

21. Code Quality Requirements

The generated code MUST be:

Style

* modular
* reproducible
* type-hinted
* documented

⸻

Performance

Must support:

* multi-GPU
* DDP
* num_workers
* pin_memory

⸻

22. Optional Enhancements

The LLM MAY additionally implement:

* packed-sequence masking
* feature engineering
* temporal sorting
* stochastic pulse sampling
* EMA
* SWA
* pseudo labeling

⸻

23. Final Goal

The generated pipeline MUST include:

* dataset
* preprocessing
* models
* losses
* trainer
* inference
* ensemble
* submission generation

Target performance:

CV ≈ 0.98

** Avoid hardcoded assumptions in data pipelines: filename parsers, dataset splits, scaler fitting logic, and submission formats must remain globally consistent and extension-agnostic, otherwise training may crash before validation and silently introduce data leakage.