DOG BREED IDENTIFICATION — GOLD-STABLE BLUEPRINT

(Kaggle Style High-Rank Solution Architecture)

比赛：Dog Breed Identification

这是一个非常典型的：
细粒度分类（fine-grained classification）
问题。

真正高分方案的核心不是：

* “更深网络”
* “更大batch”
* “更复杂augmentation”

而是：

核心本质

高质量预训练 + 高分辨率 + 稳定集成 + label smoothing + test-time augmentation + fold diversity

这个比赛非常适合：

* CNN工程能力
* 预训练迁移
* ensemble
* pseudo label
* calibration

也非常适合让 LLM 自动生成大量 diversified pipelines。

⸻

1. TOP SOLUTION DESIGN

顶级方案通常：

Stage A — Strong Single Models

训练大量：

EfficientNet 系

* B3
* B4
* B5
* B6

SE-ResNeXt

* se_resnext50_32x4d
* se_resnext101_32x4d

DenseNet

* densenet201

Inception family

* inception_v4
* inception_resnet_v2

NASNet

* nasnetalarge

后期 modernizable

今天甚至可替换成：

* ConvNeXt
* EVA
* ViT
* Swin
* MaxViT

但 Kaggle old-era 最稳定的仍是：
EfficientNet + SE-ResNeXt。

⸻

2. IMAGE SIZE STRATEGY（极重要）

这个比赛：
resolution 非常关键。

狗品种差异：

* 耳朵
* 毛发纹理
* 嘴部
* 眼睛
* 毛色 pattern

都属于：
fine-grained micro-features。

因此：

Progressive Resolution

Stage1

224

Stage2

320

Stage3

448 / 512

最后 ensemble。

⸻

3. DATA SPLIT（核心）

很多人会犯：

泄漏问题

因为：
同一只狗可能多个姿态。

必须：

Stratified KFold

通常：

StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

⸻

4. AUGMENTATION（关键）

这个比赛 augmentation 收益非常大。

⸻

必须有

Geometry

HorizontalFlip
ShiftScaleRotate
RandomResizedCrop

⸻

Color

RandomBrightnessContrast
HueSaturationValue
RGBShift

⸻

Regularization

CoarseDropout
Cutout
RandomErasing

⸻

5. LOSS DESIGN

⸻

baseline

CrossEntropyLoss

⸻

高分方案

Label Smoothing

非常有效。

label_smoothing=0.1

原因：

狗品种很多相似类别：

* husky
* malamute
* eskimo dog

硬标签会过拟合。

⸻

更进一步

Focal Loss

后期可用于 difficult breeds。

但：
多数金牌方案最终仍然：
CE + smoothing。

⸻

6. OPTIMIZER

⸻

最稳定

AdamW

⸻

lr

典型：

3e-4

⸻

scheduler

Cosine

CosineAnnealingLR

或者：

OneCycleLR

⸻

7. MIXUP / CUTMIX（非常重要）

这个比赛：
Mixup 收益巨大。

因为：
很多类别视觉近似。

⸻

Mixup

alpha = 0.2~0.4

⸻

CutMix

后期更强。

因为局部区域：
耳朵/鼻子/毛色。

⸻

8. TRAINING RECIPE（核心）

⸻

FP16

必须。

⸻

EMA

非常推荐：

ema_decay = 0.999

⸻

Gradient Clipping

clip_grad_norm_ = 1.0

⸻

Accumulation

高分辨率时：

accum_steps = 2~8

⸻

9. TEST TIME AUGMENTATION（超关键）

这个比赛：

TTA收益巨大

因为：
狗姿态变化大。

⸻

标准 TTA

original
horizontal flip
center crop
slightly scaled

通常：

4~8 TTA。

⸻

10. ENSEMBLE（真正决定排名）

这个比赛：

ensemble > 单模型

非常明显。

⸻

Winning Strategy

不是：
一个超级模型。

而是：

大量中强模型平均

例如：

Model	LB
B4	0.16
SE-RX101	0.15
DenseNet201	0.17

ensemble：

→ 0.11

⸻

11. PSEUDO LABEL（后期核心）

Public LB 很稳定。

因此：

pseudo label 非常有效

流程：

Step1

训练基础模型

Step2

生成 test soft labels

Step3

选择：

confidence > 0.95

Step4

重新训练。

⸻

12. CALIBRATION（隐藏大杀器）

这个比赛 metric：

multi-class log loss

不是 accuracy。

所以：

概率质量极重要。

⸻

Temperature Scaling

特别有效。

很多队伍：
最后仅 calibration 就提升明显。

⸻

13. TOP-LEVEL ARCHITECTURE

真正强方案：

5 folds
×
6 architectures
×
2 resolutions
×
TTA
×
Pseudo Label
×
Calibration

最终：

60~100 checkpoints ensemble

非常常见。

⸻

14. WHAT ACTUALLY MATTERS

真正重要程度：

Component	Importance
pretrained backbone	10
resolution	10
ensemble	10
TTA	9
label smoothing	8
augmentation	8
pseudo label	8
calibration	9
optimizer tricks	4
fancy losses	3

⸻

15. MODERN 2026 UPGRADE PATH

如果今天重新打：

推荐：

⸻

Tier1 backbone

ConvNeXtV2

EVA02

SwinV2

ViT-L

⸻

Training

timm

核心：

timm￼

⸻

Aug

albumentations

Albumentations￼

⸻

Loss

SoftTargetCrossEntropy

⸻

Strong tricks

SAM

EMA

Mixup

CutMix

⸻

16. LLM-GENERATABLE BLUEPRINT（最重要部分）

下面是：

真正适合 Agent/LLM 自动生成的结构

⸻

MASTER PIPELINE

configs/
    effb4_448.yaml
    seresnext101.yaml
    convnext.yaml
datasets/
    dataset.py
    augment.py
models/
    factory.py
    losses.py
training/
    train_fold.py
    ema.py
    mixup.py
inference/
    tta.py
    infer.py
ensemble/
    average.py
    calibrate.py

⸻

17. LLM TASK DECOMPOSITION

你应该让 agent：

⸻

Agent A — Backbone Generator

自动生成：

* EfficientNet
* ConvNeXt
* ViT
* Swin

训练代码。

⸻

Agent B — Augmentation Search

自动搜索：

* crop ratio
* color jitter
* erasing

⸻

Agent C — Fold Trainer

负责：

* OOF
* EMA
* AMP
* checkpoint

⸻

Agent D — Ensemble Search

自动搜索：

best_weights

例如：

0.3*b4
+0.25*convnext
+0.45*vit

优化 logloss。

⸻

Agent E — Calibration

自动：

* temperature scaling
* classwise scaling

⸻

18. FINAL GOLD RECIPE

如果你现在让 LLM 直接生成：

推荐：

⸻

Backbone

ConvNeXt Base
EfficientNet B5
SE-ResNeXt101

⸻

Resolution

384 + 448

⸻

Loss

CE + Label Smoothing

⸻

Tricks

EMA
Mixup
CutMix
TTA
Pseudo Label

⸻

Ensemble

20~40 models

⸻

19. 最后的关键认知

这个比赛不是：

“谁模型最强”

而是：

“谁的概率最稳定”

因为：

metric 是：

multi-class log loss

因此：

很多时候：

* calibration
* ensemble diversity
* TTA stability

比单模型 accuracy 更重要。

⸻
FORBIDDEN RULE — Dynamic Class Count Inference

Critical Constraint

When implementing:

* MixUp
* CutMix
* Label Smoothing
* Soft Target Cross Entropy
* Any soft-label pipeline

the implementation MUST NEVER infer the number of classes dynamically from the current minibatch.

⸻

STRICTLY FORBIDDEN PATTERNS

The following patterns are PROHIBITED:

labels.max() + 1
int(labels.max().item() + 1)
len(torch.unique(labels))
classes = num_classes or labels.max()+1
classes = getattr(criterion, "num_classes", None)
classes = criterion.num_classes

when criterion is nn.CrossEntropyLoss.

⸻

WHY THIS IS FORBIDDEN

A minibatch is NOT guaranteed to contain:

* all dataset classes
* the maximum class index

Example:

Dataset classes: 120
Batch max label: 117

Incorrect inference:

classes = labels.max()+1
# → 118

creates:

soft_targets.shape = [B,118]
logits.shape       = [B,120]

which causes:

RuntimeError:
The size of tensor a (118)
must match the size of tensor b (120)

⸻

REQUIRED IMPLEMENTATION

The class count MUST come ONLY from a global fixed configuration:

cfg.num_classes

or:

dataset.num_classes

and MUST be explicitly propagated through:

* train loop
* validation loop
* MixUp
* CutMix
* soft-label utilities

⸻

REQUIRED FUNCTION SIGNATURE

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    num_classes,
):

⸻

REQUIRED USAGE

classes = int(num_classes)

NOT:

classes = labels.max()+1

⸻

REQUIRED ASSERTION

Before loss computation, the implementation MUST verify:

assert logits.shape[1] == num_classes
assert soft_targets.shape[1] == num_classes

⸻

GOLD RULE

Class count is a DATASET property,
NOT a BATCH property.