GOLD-STABLE BLUEPRINT — TGS Salt Identification Challenge (30th-Place Style + TernausNet Lineage)

1. CORE PHILOSOPHY

This competition is NOT solved by:

* giant UNet alone
* brute-force TTA
* handcrafted threshold tricks
* excessive CRF/postprocessing
* public LB overfitting
* single-loss optimization

The competition is solved by:

Stable Boundary-Aware Segmentation

through:

* strong pretrained encoder-decoder architectures
* aggressive augmentation
* boundary-sensitive optimization
* hypercolumn-style decoder refinement
* careful validation protocol
* robust training scheduling
* fold consistency
* high-quality inference averaging

The key insight:

TGS Salt is a tiny-object + fuzzy-boundary + low-data segmentation problem.

The winning direction is therefore:

* preserve spatial detail
* optimize edges explicitly
* maximize generalization
* reduce decoder artifacts
* stabilize validation

⸻

2. COMPETITION CHARACTERISTICS

Dataset Nature

* binary segmentation
* image size = 101×101
* grayscale seismic images
* highly imbalanced masks
* many empty masks
* salt boundaries are ambiguous
* train set extremely small

This means:

Critical Problems

* overfitting
* unstable folds
* decoder checkerboard artifacts
* boundary collapse
* false positives on empty masks

⸻

3. TARGET ARCHITECTURE

PRIMARY MODEL FAMILY

Use:

* TernausNet / AlbuNet style UNet
* pretrained ImageNet encoder
* strong decoder refinement

Preferred encoders:

1. se_resnext50
2. se_resnet50
3. resnet152
4. resnet101
5. vgg16 (baseline)
6. vgg11

⸻

4. REQUIRED ARCHITECTURE MODIFICATIONS

4.1 REMOVE FINAL ENCODER POOLING

Very important.

The original writeup explicitly states:

“Remove that famous pooling :p”

Meaning:

DO NOT aggressively downsample deepest features.

Typical bad pipeline:

conv5 -> maxpool -> bottleneck

Preferred:

conv5 -> bottleneck

Reason:

* tiny objects
* tiny image resolution
* preserving spatial structure matters more than receptive field

⸻

5. DECODER DESIGN

DO NOT USE NAIVE DECONV

Avoid classic transposed-conv-heavy decoders.

Instead prefer:

Conv -> Upsample(Bilinear) -> Conv

NOT:

Deconv -> Conv

Reason:

* reduces checkerboard artifacts
* smoother masks
* better validation IoU

The readme explicitly notes:

“Upsample instead of Deconv”

⸻

6. HYPERCOLUMN DECODER

Strongly recommended.

Required

Fuse multi-scale decoder outputs.

Example:

hypercolumn = torch.cat([
    dec1,
    F.interpolate(dec2),
    F.interpolate(dec3),
    F.interpolate(dec4),
], dim=1)

Then:

final_conv(hypercolumn)

Benefits:

* sharper boundaries
* better tiny-object recovery
* stronger local-global fusion

⸻

7. SQUEEZE-EXCITATION IN DECODER

Very important improvement.

Add SE blocks after decoder convs.

Example:

class SCSE(nn.Module):

Use:

* channel squeeze excitation
* spatial squeeze excitation

Especially effective for:

* noisy seismic textures
* empty-mask suppression

⸻

8. DROPOUT STRATEGY

Required

Use decoder dropout:

Dropout2d(p=0.5)

ONLY in decoder/hypercolumn.

NOT encoder.

Reason:

* encoder already pretrained
* decoder massively overfits

⸻

9. LOSS DESIGN

EARLY TRAINING

Use:

BCE + 10 * BoundaryLoss

Boundary loss is critical.

Possible implementations:

* edge BCE
* contour BCE
* Laplacian boundary BCE

⸻

10. LATE TRAINING

After ~30 epochs:

Switch to:

LovaszHinge + 10 * BoundaryLoss

This is extremely important.

Reason:

* BCE optimizes pixels
* Lovasz optimizes IoU surrogate
* TGS metric is IoU-threshold based

⸻

11. BOUNDARY TARGET GENERATION

Generate edge masks from GT masks.

Typical method:

boundary = dilation(mask) - erosion(mask)

or Sobel/Laplacian.

Train auxiliary boundary head.

⸻

12. INPUT PIPELINE

Replicated Padding

The writeup explicitly mentions:

“Backprop on full image (with replicated borders included)”

Required pipeline:

101x101
→ replicate pad
→ 128x128

NOT zero padding.

Reason:

* seismic borders contain structure
* zero padding introduces artifacts

⸻

13. AUGMENTATION POLICY

HARD AUGMENTATIONS WORK BEST

This is unusual but very important.

Use:

* brightness
* contrast
* gamma
* blur
* horizontal flip
* elastic shift
* scale
* rotation
* affine
* random crop
* shift up to 50 pixels

Aggressive augmentation is necessary because:

* dataset tiny
* seismic texture highly variable

⸻

14. FINAL EPOCH STRATEGY

Very important detail from writeup:

“Last 10 epochs with no augmentations helped”

Final stage:

ONLY:

* horizontal flip
* light normalization

Disable:

* heavy distortions
* blur
* geometric warping

Reason:

* helps final convergence
* reduces train/inference mismatch

⸻

15. OPTIMIZER

Preferred:

AdamW

or classic:

Adam

Initial LR:

1e-4

Final LR:

1e-6

⸻

16. LR SCHEDULE

Recommended:

Stage 1

StepLR(gamma=0.5)

Stage 2

Switch to:

ReduceLROnPlateau

This mirrors the original writeup.

⸻

17. VALIDATION STRATEGY

ABSOLUTELY CRITICAL

Use:

Stratified KFold

Stratify by:

* empty/non-empty masks
* coverage class

Typical:

coverage = mask.sum() / (101*101)
coverage_class = coverage_to_class(coverage)

Use:

5 folds

⸻

18. VALIDATION METRIC

Evaluate on:

cropped 101x101 masks

NOT padded masks.

This matters.

⸻

19. CHECKPOINT SELECTION

Very important hidden trick from writeup:

“Save checkpoint on avg validation between val and val_tta”

Meaning:

Checkpoint metric should be:

0.5 * val_score + 0.5 * val_tta_score

NOT plain validation only.

This improves inference robustness.

⸻

20. TTA

Use conservative TTA only.

Recommended:

* horizontal flip
* small scale TTA

Avoid:

* heavy rotation TTA
* complicated geometric TTA

because:

* seismic orientation matters

⸻

21. EMPTY MASK HANDLING

Critical.

Add:

Classification Auxiliary Head

Predict:

is_empty

from bottleneck features.

Use during inference:

if empty_prob > threshold:
    mask = 0

Massively reduces FP masks.

⸻

22. POSTPROCESSING

Minimal postprocessing preferred.

Do NOT rely heavily on:

* CRF
* morphology
* heuristic hole filling

Simple:

remove_small_objects

is enough.

⸻

23. PSEUDOLABELING

The writeup explicitly says:

“Pseudolabeling”

Strongly recommended.

Pipeline:

1. Train folds
2. Infer test
3. Keep high-confidence masks
4. Retrain with pseudo labels

Use only:

confidence > 0.9

⸻

24. ENSEMBLE STRATEGY

Strong final solutions usually use:

* multiple folds
* multiple seeds
* multiple encoders

Preferred diversity:

* se_resnext50
* resnet152
* densenet
* efficientnet

⸻

25. INFERENCE DETAILS

Threshold Search

Tune:

* binarization threshold
* minimum component size

using OOF only.

Never tune on leaderboard.

⸻

26. RECOMMENDED FINAL ARCHITECTURE

GOLD-STABLE RECIPE

Encoder

se_resnext50_32x4d

Decoder

* bilinear upsample
* hypercolumns
* scSE blocks
* dropout

Loss

LovaszHinge + Boundary BCE

Training

* 5 folds
* AdamW
* heavy augmentation
* pseudo labels

⸻

27. IMPORTANT IMPLEMENTATION RULES FOR LLM CODE GENERATION

MUST DO

Architecture

* use pretrained encoder
* use bilinear upsampling
* remove deepest pooling
* use skip connections
* use hypercolumns
* use scSE blocks

Training

* mixed precision
* gradient clipping
* fold-safe validation
* deterministic seeds
* OOF prediction saving

Loss

* lovasz hinge
* BCE
* boundary-aware auxiliary loss

Inference

* flip TTA
* averaged folds
* threshold search

⸻

28. FORBIDDEN DESIGN CHOICES

NEVER DO THESE

Architecture

* naive UNet from scratch
* transpose-conv-only decoder
* no pretrained encoder
* no skip connections

Training

* random train/val split
* no stratification
* training only BCE
* training only Dice
* excessive augmentation in final epochs

Validation

* selecting checkpoint on train loss
* evaluating padded masks
* LB-based threshold tuning

⸻

29. EXPECTED PERFORMANCE TIERS

Basic UNet

~0.82 - 0.84

TernausNet/VGG16

~0.84 - 0.86

ResNet + Lovasz + Boundary

~0.86 - 0.88

Full optimized pipeline

gold-level capable

⸻

30. FINAL META-LESSON

TGS Salt is fundamentally a:

Boundary Preservation Problem

NOT a pure segmentation problem.

The strongest solutions succeed because they:

* preserve spatial detail
* stabilize tiny datasets
* optimize boundaries directly
* minimize decoder artifacts
* use robust validation discipline

The combination of:

* pretrained encoder
* hypercolumn decoder
* Lovasz optimization
* boundary supervision
* strong augmentation
* careful validation

is what creates a truly competitive TGS pipeline.