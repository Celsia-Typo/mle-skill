# SCoT Skill: Paddy Disease Classification Competition

---

## Instructions

You are an expert Kaggle competitor specializing in computer vision for agricultural disease classification. Your goal is to write a complete, runnable Python training pipeline for the **Paddy Disease Classification** competition (10-class image classification on 16,225 rice paddy samples, scored by accuracy).

**First, sketch a rough problem-solving process using three programming structures (sequential, branch, and loop structures), then output the final code.**

Background knowledge (derived from verified experiments):
- All best solutions use **ConvNeXt from timm** with **ImageNet-22k pretrained weights**
- The biggest single gain: switching to IN22k pretraining → +0.00461
- Second biggest gain: K-Fold cross-validation ensemble → +0.00193
- A confirmed regression (−0.008) was caused by a scheduler granularity bug (see Critical Constraint #1 below)
- A confirmed TTA redundancy (5th view = duplicate of 1st) silently biases ensemble toward original image
- Current best score: 0.98002 (ConvNeXt-base, 3-fold, 224px); Bronze threshold: 0.98387
- **lr_backbone underfitting evidence**: ConvNeXtV2-Large with `lr_backbone=2e-5`, 5-fold, 384px, 20 epochs → OOF 0.9721, train loss still 0.91 at epoch 20 (confirmed underfitting). Raising to `lr_backbone=5e-5` is the primary fix.
- **Two-phase training rationale**: single-phase 384px at 20 epochs hits the 4h H100 time limit with no room to add epochs; 224px Phase 1 (~0.9h) + 384px Phase 2 (~2.4h) delivers more convergence in ≤3.5h total.
- **Epoch reduction is always wrong**: reducing epochs 20→10 to fix timeouts drops OOF by −0.043 (0.9721→0.9393, confirmed); the correct fix is API compliance + NW=2 (see Critical Constraint #5).
- **Single-model ceiling**: estimated ~0.982–0.985 for ConvNeXtV2-Large alone; exceeding Bronze (0.98387) reliably requires cross-model ensemble (see Post-Training section).

Constraints and goals:
- Target: exceed **0.98387** (Bronze); aim for **0.98617** (Silver)
- Hardware: single NVIDIA H100 80GB, 24-hour budget
- Preferred backbone: **`convnextv2_large.fcmae_ft_in22k_in1k_384`** (ConvNeXt V2 with FCMAE pretraining; fallback: `convnext_large.fb_in22k_ft_in1k_384`)
- Must use `timm` pretrained backbone, `albumentations` for augmentation, `AdamW` optimizer
- Must include: K-Fold CV (≥5 folds), AMP mixed-precision, layerwise learning rates, EMA, CLAHE augmentation, TTA×5 (all 5 views must be genuinely distinct), soft-prob ensemble across folds
- Output: `submission.csv` with columns `[image_id, label]` and `.npy` soft probs for cross-model ensembling

**⚠️ CRITICAL CONSTRAINT #1 — Scheduler Step Granularity:**

The LR scheduler MUST be stepped **once per batch** (inside the training loop), NOT once per epoch. Mixing batch-level `T_max` with epoch-level `step()` silently freezes the LR (confirmed −0.008 regression):

```python
# CORRECT — step-level: scheduler.step() lives INSIDE the batch for-loop
total_steps  = epochs * len(train_loader)
scheduler    = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
for epoch in range(epochs):
    for imgs, targets in train_loader:
        ...optimizer.step()...
        scheduler.step()   # ← INSIDE batch loop

# WRONG — epoch-level step() with batch-level T_max (LR barely moves)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
for epoch in range(epochs):
    train_one_epoch(...)
    scheduler.step()       # ← outside batch loop = silent bug
```

**⚠️ CRITICAL CONSTRAINT #2 — TTA Views Must Be Genuinely Distinct:**

All 5 TTA augmentation views must produce different image variants. Using `get_val_transform()` (plain resize) as one of the 5 views duplicates the "original" view, silently biasing the ensemble. Use exactly these 5:

```python
# CORRECT — 5 genuinely distinct views
tta_views = [orig, hflip, vflip, hflip+vflip, rot90]

# WRONG — 5th view is a duplicate of 1st (confirmed redundancy in best_solution01.py)
tta_views = [orig, hflip, vflip, hflip+vflip, val_resize]  # val_resize == orig
```

**⚠️ CRITICAL CONSTRAINT #3 — No `deterministic=True` on H100:**

```python
# CORRECT — enables cuDNN auto-tuning, ~20-30% faster training on H100
torch.backends.cudnn.benchmark    = True
torch.backends.cudnn.deterministic = False

# WRONG — disables cuDNN optimizations, wastes ~4-6 epochs worth of H100 compute
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark    = False
```

**⚠️ CRITICAL CONSTRAINT #4 — Only One Training Function (AMP version):**

Do NOT define a non-AMP `train_one_epoch` alongside `train_one_epoch_amp`. Dead non-AMP code that lacks `scheduler.step()` causes confusion and silent bugs if mistakenly called. Define only one training function that includes: AMP autocast, scaler, scheduler.step() per batch, and EMA update.

**⚠️ CRITICAL CONSTRAINT #5 — Environment API Compatibility (prevents silent timeouts):**

The execution environment has specific API requirements. Violating these causes deprecation warnings that flood stdout, slow training, and may trigger incorrect timeout-avoidance fixes (e.g., reducing epochs from 20 to 10, which drops OOF from 0.9721 to 0.9393). Use the correct APIs:

```python
# ── Albumentations (new API — old kwarg names are deprecated and noisy) ────
# CORRECT
A.GaussNoise(p=1.0)                                      # no var_limit kwarg
A.CoarseDropout(
    num_holes_range=(2, 8),
    hole_height_range=(img_size // 24, img_size // 12),
    hole_width_range=(img_size // 24, img_size // 12),
    p=0.3)
# WRONG (deprecated, produces many FutureWarning logs)
A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
A.CoarseDropout(max_holes=8, max_height=..., max_width=..., min_holes=2, p=0.3)

# ── PyTorch AMP (new API — torch.cuda.amp.* is deprecated) ────────────────
# CORRECT
with torch.amp.autocast('cuda'):
    ...
scaler = torch.amp.GradScaler('cuda')
# WRONG (deprecated since PyTorch 2.x)
with torch.cuda.amp.autocast():
    ...
scaler = torch.cuda.amp.GradScaler()

# ── DataLoader workers (system limit is 3; exceeding it causes hangs) ──────
# CORRECT
NW = 2   # or: min(2, os.cpu_count() - 1)
# WRONG
NW = 4   # or NW = 8 — exceeds system limit, causes DataLoader warnings

# ── torch.load (security best practice) ───────────────────────────────────
# CORRECT
ckpt = torch.load(path, map_location=device, weights_only=True)
# WRONG
ckpt = torch.load(path, map_location=device)  # weights_only defaults to False
```

**DO NOT reduce `epochs` below 20 to fix timeout issues.** If training is too slow, the root cause is almost always API warnings (albumentations / torch.amp deprecations) or too many DataLoader workers — fix those instead. Reducing epochs from 20→10 caused a confirmed −0.043 OOF regression (0.9721 → 0.9393).

**⚠️ CRITICAL CONSTRAINT #6 — Two-Phase Training (224px → 384px):**

Single-phase 384px training at 20 epochs exhausts the 4h H100 budget with the model still underfitting (train loss 0.91 at epoch 20). Two-phase training resolves this by spending Phase 1 at lower resolution for fast convergence, then Phase 2 at full resolution for fine detail — fitting within ~3.5h total:

```
Phase 1: img_size=224, epochs=12, batch=32  → ~0.9h  (backbone fast adaptation)
Phase 2: img_size=384, epochs=12, batch=16  → ~2.4h  (high-res fine-tuning)
Total  :                                       ~3.3h  (0.7h headroom before 4h limit)
```

```python
# ── CORRECT: two-phase training per fold ─────────────────────────────────
def train_fold_two_phase(model, train_df, val_df, fold, cfg):
    # Phase 1 — 224px, higher batch, more batches/epoch → faster LR cycling
    loader_224 = build_loader(train_df, img_size=224, batch=32, augment=True)
    total_p1   = cfg.epochs_p1 * len(loader_224)
    opt, sch   = build_optimizer_and_scheduler(model, total_p1,
                     lr_backbone=5e-5, lr_head=5e-4)
    for ep in range(cfg.epochs_p1):          # e.g. 12 epochs
        train_one_epoch_amp(model, loader_224, opt, sch, ...)

    # Phase 2 — resume same model weights, switch to 384px
    loader_384 = build_loader(train_df, img_size=384, batch=16, augment=True)
    total_p2   = cfg.epochs_p2 * len(loader_384)
    opt2, sch2 = build_optimizer_and_scheduler(model, total_p2,
                     lr_backbone=2e-5, lr_head=2e-4)  # lower LR for fine-tuning
    for ep in range(cfg.epochs_p2):          # e.g. 12 epochs
        train_one_epoch_amp(model, loader_384, opt2, sch2, ...)

    # TTA inference always at 384px regardless of training phase
    tta_preds = run_tta(model, test_loader_384, tta_transforms_384)
    return tta_preds

# ── WRONG: single-phase 384px hits 4h wall with train_loss still 0.91 ─────
# for ep in range(20):
#     train_one_epoch_amp(model, loader_384, ...)   # never converges in time
```

Key rule: **TTA inference at the end of Phase 2 must use 384px** (the high-resolution views), regardless of which phase the model was most recently trained on.

---

## Demonstration Examples

### Example 1

**Requirement:**

```python
def build_optimizer_and_scheduler(
        model: nn.Module,
        total_steps: int,
        lr_backbone: float = 5e-5,
        lr_head: float = 5e-4,
        weight_decay: float = 0.05,
        warmup_frac: float = 0.06) -> tuple:
    """
    Build AdamW with layerwise LRs and a step-level warmup-cosine scheduler.
    IMPORTANT: the returned scheduler must be stepped once per BATCH (not per epoch).

    Args:
        model: timm ConvNeXt/ConvNeXtV2 model with a "head" module
        total_steps: total optimizer steps = epochs × len(train_loader)
        lr_backbone: LR for backbone layers. Use 5e-5 (NOT 2e-5 — confirmed underfitting
                     at 2e-5: OOF 0.9721 vs baseline 0.98002 despite stronger model).
        lr_head: LR for classification head (10× backbone for fast task adaptation)
        weight_decay: AdamW weight decay
        warmup_frac: fraction of total_steps for linear warmup (default 6%)

    NOTE on model creation: always pass drop_path_rate to timm.create_model for
    stochastic depth regularization on large backbones:
        model = timm.create_model(model_name, pretrained=True,
                                  num_classes=num_classes, drop_path_rate=0.2)

    Returns:
        (optimizer, scheduler): AdamW and step-level LambdaLR
    """
```

**SCoT → Code:**

```python
def build_optimizer_and_scheduler(
        model: nn.Module,
        total_steps: int,
        lr_backbone: float = 5e-5,   # 2e-5 confirmed underfitting (OOF 0.9721); use 5e-5
        lr_head: float = 5e-4,       # 10× backbone for fast head adaptation
        weight_decay: float = 0.05,
        warmup_frac: float = 0.06) -> tuple:
    import math
    from torch.optim.lr_scheduler import LambdaLR

    # ── SEQUENTIAL: split parameters into backbone / head groups ──────
    HEAD_KEYWORDS = ("head", "classifier", "fc")
    head_params, backbone_params = [], []

    # ── LOOP: iterate over named parameters ───────────────────────────
    for name, param in model.named_parameters():
        # ── BRANCH: assign by keyword match ──────────────────────────
        if any(kw in name for kw in HEAD_KEYWORDS):
            head_params.append(param)
        else:
            backbone_params.append(param)

    # ── SEQUENTIAL: build AdamW with two LR groups ────────────────────
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone,
         "weight_decay": weight_decay},
        {"params": head_params,     "lr": lr_head,
         "weight_decay": weight_decay * 0.2},
    ])

    # ── SEQUENTIAL: build step-level warmup-cosine scheduler ──────────
    # MUST be stepped once per batch inside the training loop (not per epoch)
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        # ── BRANCH: warmup phase (linear ramp) ───────────────────────
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        # ── BRANCH: cosine decay phase ────────────────────────────────
        progress = float(step - warmup_steps) / \
                   float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
    # Let's think step by step — write your code here
```

---

### Example 2

**Requirement:**

```python
def get_train_transform(img_size: int) -> A.Compose:
    """
    Training augmentation pipeline for paddy disease classification.
    Includes CLAHE for disease lesion contrast enhancement and GridDistortion
    for leaf shape variation — both validated on agricultural image datasets.
    All geometric flips/rotations are applied before pixel-level transforms.

    Args:
        img_size: target square crop size (e.g. 384)

    Returns:
        albumentations Compose pipeline
    """
```

**SCoT → Code:**

```python
def get_train_transform(img_size: int) -> A.Compose:
    # ── SEQUENTIAL: define augmentation stages in order ───────────────
    return A.Compose([
        # Stage 1: geometric crop (random scale + aspect)
        A.RandomResizedCrop(
            size=(img_size, img_size), scale=(0.65, 1.0), ratio=(0.75, 1.33)
        ),
        # Stage 2: geometric flips and rotation
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),

        # Stage 3: disease-specific pixel transforms
        # CLAHE enhances local contrast to highlight disease lesions
        # (e.g. hispa silver streaks, blast diamond spots, brown_spot halos)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
        # GridDistortion simulates natural leaf shape deformation
        A.GridDistortion(num_steps=5, distort_limit=0.25, p=0.25),

        # Stage 4: color jitter
        A.RandomBrightnessContrast(
            brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=12, p=0.4),

        # Stage 5: noise / blur (one of three, applied 30% of the time)
        # NOTE: GaussNoise must NOT use var_limit kwarg (deprecated API)
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        ], p=0.3),

        # Stage 6: regularization via random erasing
        # NOTE: CoarseDropout must use new API (max_holes/max_height deprecated)
        A.CoarseDropout(
            num_holes_range=(2, 8),
            hole_height_range=(img_size // 24, img_size // 12),
            hole_width_range=(img_size // 24, img_size // 12),
            p=0.3),

        # Stage 7: normalize to ImageNet stats, convert to tensor
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    # Let's think step by step — write your code here


def get_val_transform(img_size: int) -> A.Compose:
    # ── SEQUENTIAL: deterministic resize + normalize only ─────────────
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int) -> list:
    """
    Return 5 GENUINELY DISTINCT TTA augmentation pipelines.
    Do NOT include val_transform (plain resize) as one of the 5 views —
    it duplicates view 0 (original) and silently biases the ensemble.
    """
    # ── SEQUENTIAL: define 5 distinct views ───────────────────────────
    norm = [A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()]
    return [
        A.Compose([A.Resize(img_size, img_size)] + norm),                      # 0: original
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.)] + norm), # 1: h-flip
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.)]   + norm), # 2: v-flip
        A.Compose([A.Resize(img_size, img_size),                                   # 3: h+v flip
                   A.HorizontalFlip(p=1.), A.VerticalFlip(p=1.)]         + norm),
        A.Compose([A.Resize(img_size, img_size),                                   # 4: rot90
                   A.RandomRotate90(p=1.)]                                + norm),
        # View 4 uses RandomRotate90 (deterministic 90° in albumentations)
        # Do NOT replace view 4 with val_transform — that would duplicate view 0
    ]
    # Let's think step by step — write your code here
```

---

### Example 3

**Requirement:**

```python
def train_one_epoch_amp(model: nn.Module,
                        loader: DataLoader,
                        optimizer: torch.optim.Optimizer,
                        scheduler,
                        criterion: nn.Module,
                        mixup_fn,
                        scaler: torch.amp.GradScaler,
                        ema,
                        device: str = "cuda",
                        grad_clip: float = 1.0) -> float:
    """
    Single training epoch: AMP + MixUp/CutMix + per-batch scheduler step + EMA.
    This is the ONLY training function — do not define a parallel non-AMP version.
    scheduler.step() is called once per batch (inside this function).

    Args:
        model: PaddyModel on device
        loader: training DataLoader (integer hard labels)
        optimizer: AdamW (layerwise LRs)
        scheduler: step-level LambdaLR — stepped inside this function per batch
        criterion: SoftTargetCrossEntropy (timm.loss)
        mixup_fn: timm Mixup (handles MixUp + CutMix jointly, outputs soft targets)
        scaler: torch.amp.GradScaler('cuda') — use new API, NOT torch.cuda.amp.GradScaler()
        ema: object with .update(model) method
        device: torch device string
        grad_clip: gradient clipping max norm

    Returns:
        avg_loss: mean training loss for this epoch
    """
```

**SCoT → Code:**

```python
def train_one_epoch_amp(model: nn.Module,
                        loader: DataLoader,
                        optimizer: torch.optim.Optimizer,
                        scheduler,
                        criterion: nn.Module,
                        mixup_fn,
                        scaler: torch.amp.GradScaler,   # torch.amp.GradScaler('cuda')
                        ema,
                        device: str = "cuda",
                        grad_clip: float = 1.0) -> float:
    # ── SEQUENTIAL: set train mode, init accumulators ─────────────────
    model.train()
    total_loss, total_n = 0.0, 0

    # ── LOOP: iterate over batches ────────────────────────────────────
    for imgs, targets in loader:
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # ── SEQUENTIAL: MixUp/CutMix → soft one-hot targets ──────────
        # timm Mixup randomly selects MixUp or CutMix per batch;
        # label_smoothing is baked into the soft targets here, so
        # use SoftTargetCrossEntropy (not CrossEntropyLoss) as criterion
        imgs, soft_targets = mixup_fn(imgs, targets)

        # ── SEQUENTIAL: zero grad (set_to_none releases memory) ───────
        optimizer.zero_grad(set_to_none=True)

        # ── SEQUENTIAL: AMP forward + loss ────────────────────────────
        # NOTE: use torch.amp.autocast('cuda'), NOT torch.cuda.amp.autocast()
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss   = criterion(logits, soft_targets)

        # ── SEQUENTIAL: AMP backward + grad clip + optimizer step ─────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # ── SEQUENTIAL: step scheduler PER BATCH (step-level) ─────────
        # CRITICAL: this line is inside the for-loop.
        # Moving it outside (per epoch) silently freezes LR (−0.008 bug).
        scheduler.step()

        # ── SEQUENTIAL: update EMA weights each batch ─────────────────
        ema.update(model)

        total_loss += loss.item() * imgs.size(0)
        total_n    += imgs.size(0)

    return total_loss / total_n
    # Let's think step by step — write your code here
```

---

## Testing Requirement

Write a **complete end-to-end script** that fixes all confirmed bugs in `best_solution01.py` (score=0.97194) and exceeds Bronze (0.98387):

```python
def main_upgrade_pipeline(data_dir: str = "./input",
                           output_dir: str = "./submission",
                           model_name: str = "convnextv2_large.fcmae_ft_in22k_in1k_384",
                           img_size: int = 384,
                           n_folds: int = 5,
                           epochs: int = 20,
                           device: str = "cuda") -> str:
    """
    Medal-targeting pipeline for Paddy Disease Classification.

    Fixes applied vs best_solution01.py (score=0.97194):
      [FIX 1] scheduler.step() moved INSIDE the batch loop (was: per epoch → LR frozen)
      [FIX 2] T_max = total_steps used consistently with per-batch step()
      [FIX 3] warmup added (6% of total_steps, linear ramp)
      [FIX 4] TTA view 5 changed from val_resize (duplicate) to rot90 (genuinely distinct)
      [FIX 5] benchmark=True, deterministic=False (was: reversed → 20-30% slower)
      [FIX 6] Only one training function (AMP version); non-AMP dead code removed
      [FIX 7] optimizer.zero_grad(set_to_none=True) throughout
      [FIX 8] torch.amp.autocast('cuda') / torch.amp.GradScaler('cuda') — new API
      [FIX 9] A.GaussNoise(p=1.0), A.CoarseDropout(num_holes_range=...) — new albumentations API
      [FIX 10] NW = 2 (was: NW=4/8, exceeded system limit → DataLoader warnings + slowdowns)
      [FIX 11] torch.load(..., weights_only=True)
      [FIX 12] output_dir defaults to ./submission (not ./submission_upgraded)

    Upgrades vs best_solution03.py (score=0.98002):
      [UP 1] convnextv2_large (FCMAE+IN22k+IN1k) > convnext_base (IN1k)
      [UP 2] Two-phase training: 224px×12ep (~0.9h) → 384px×12ep (~2.4h), total ~3.3h
      [UP 3] 5-Fold > 3-Fold
      [UP 4] CLAHE + GridDistortion augmentation added
      [UP 5] EMA used for validation and final TTA inference
      [UP 6] Saves ensemble_test_probs.npy for cross-model ensembling
      [UP 7] lr_backbone=5e-5 (was 2e-5 → confirmed underfitting, OOF 0.9721)
      [UP 8] drop_path_rate=0.2 in timm.create_model for stochastic depth regularization
      [UP 9] MixUp alpha=0.4 (was 0.8 default → too aggressive for ~8k images/fold,
             needs 30+ epochs to converge; 0.4 converges within 12-epoch Phase 2)

    CRITICAL rules (do not violate):
      - scheduler.step() must be called inside the batch loop in train_one_epoch_amp
      - TTA must use 5 genuinely distinct views: orig, hflip, vflip, hflip+vflip, rot90
      - TTA inference always at 384px (even when Phase 1 trained at 224px)
      - torch.backends.cudnn.benchmark=True, deterministic=False
      - Only define train_one_epoch_amp (AMP version); do not define a non-AMP variant
      - Use timm.data.Mixup(mixup_alpha=0.4, cutmix_alpha=1.0) with SoftTargetCrossEntropy
      - model = timm.create_model(..., drop_path_rate=0.2)
      - Use torch.amp.autocast('cuda') and torch.amp.GradScaler('cuda') (NOT torch.cuda.amp.*)
      - Use A.GaussNoise(p=1.0) and A.CoarseDropout(num_holes_range=..., ...) (new albumentations API)
      - Set NW = 2 (DataLoader num_workers); system limit is 3, NW≥4 causes slowdowns
      - Use torch.load(path, map_location=device, weights_only=True)
      - DO NOT reduce epochs below 12 per phase to fix timeouts; fix API warnings + NW instead

    Args:
        data_dir: root with train.csv, train_images/, test_images/, sample_submission.csv
        output_dir: destination for submission CSV and per-fold .npy prob arrays
        model_name: timm model id (ConvNeXtV2 preferred)
        img_size: training and inference resolution
        n_folds: StratifiedKFold splits
        epochs: training epochs per fold
        device: torch device string

    Returns:
        submission_path: str, path to the final submission CSV
    """
```

**Let's think step by step. Write your code here.**

---

## Post-Training: Cross-Model Ensemble ⚠️ MANDATORY for Bronze

**Single-model ceiling**: ConvNeXtV2-Large alone is estimated to top out at ~0.982–0.985. Bronze requires ≥0.98387. The gap can only be reliably closed by ensembling two complementary backbones.

**Recommended two-run schedule (H100, fits in ~5.1h total):**

| Run | Config | Est. time | Saves |
|-----|--------|-----------|-------|
| Run 1 | ConvNeXtV2-Large, two-phase 224→384, 5-fold | ~3.3h | `./sub_cvnxt/ensemble_test_probs.npy` |
| Run 2 | `swin_large_patch4_window12_384_in22k`, 224px, 5-fold, 20ep | ~1.7h | `./sub_swinl/ensemble_test_probs.npy` |
| Ensemble | soft-voting of the two .npy files | <5 min | `submission_ensemble.csv` |

After both runs complete, combine their saved `.npy` files. This step alone adds +0.003–0.006:

```python
def cross_model_ensemble(npy_paths: list,
                          val_accs: list,
                          test_image_ids: list,
                          label_map: dict,
                          output_csv: str = "submission_ensemble.csv") -> str:
    """
    Weighted soft-voting ensemble across multiple model .npy prob arrays.
    Weight each model by its OOF val accuracy (normalized to sum to 1).

    Args:
        npy_paths: list of paths to ensemble_test_probs.npy files (one per backbone)
        val_accs: list of float OOF val accuracies, one per npy file
        test_image_ids: list of image_id strings in test set order
        label_map: dict mapping class_idx (int) → class_name (str)
        output_csv: path to write final submission CSV

    Returns:
        output_csv path

    Example:
        cross_model_ensemble(
            npy_paths  = ["./sub_convnextv2/ensemble_test_probs.npy",
                          "./sub_swinl/ensemble_test_probs.npy"],
            val_accs   = [0.988, 0.985],
            test_image_ids = test_df["image_id"].tolist(),
            label_map  = idx2label,
        )
    """
    # ── SEQUENTIAL: load all prob arrays ──────────────────────────────
    probs_list = [np.load(p) for p in npy_paths]

    # ── SEQUENTIAL: compute normalized weights from OOF val accs ──────
    w = np.array(val_accs, dtype=np.float64)
    w = w / w.sum()

    # ── SEQUENTIAL: weighted average ──────────────────────────────────
    ensemble = sum(wi * pi for wi, pi in zip(w, probs_list))

    # ── SEQUENTIAL: decode predictions and write submission ───────────
    pred_labels = [label_map[int(i)] for i in ensemble.argmax(axis=1)]
    pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})\
      .to_csv(output_csv, index=False)
    print(f"Ensemble saved → {output_csv}  weights={w.round(3)}")
    return output_csv
    # Let's think step by step — write your code here
```
