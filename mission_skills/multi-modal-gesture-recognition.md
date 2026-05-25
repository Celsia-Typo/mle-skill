Multi-Modal Gesture Recognition — Gold-Level Blueprint（LLM代码生成版）

这个方案本质上是一个：

多模态时序分割 + 序列解码 + Session级CV + Logit Ensemble

的完整工业级流水线。

目标不是“单帧分类”，而是：

* 输入：
    * Skeleton序列
    * Audio序列
* 输出：
    * Gesture sequence（如 "2 14 20"）

核心难点：

1. 多模态融合
2. 时序边界检测
3. 长序列建模
4. Session leakage防止
5. 序列级解码优化
6. Levenshtein metric 对齐

⸻

1. Overall Pipeline

Raw Archives (.tar.gz)
        ↓
Sample Extraction
        ↓
MAT + WAV Parsing
        ↓
Feature Engineering
        ↓
Session-aware KFold
        ↓
MSTC + BiGRU + Attention
        ↓
Frame-level Probabilities
        ↓
Temporal Decoding
        ↓
Gesture Sequence
        ↓
Levenshtein Evaluation
        ↓
OOF + Fold Ensemble
        ↓
Submission.csv

⸻

2. Core Philosophy

该任务本质：

Gesture Segmentation + Sequence Transcription

不是普通分类。

因此：

错误做法：

整段视频 → 单标签分类

正确做法：

Frame-wise modeling
    +
Temporal boundary learning
    +
Sequence decoding

核心目标：

最大化最终 sequence-level 编辑距离指标

而不是 frame accuracy。

⸻

3. Data Layer Blueprint

3.1 Raw Modalities

每个sample包含：

Skeleton

(T, 20 joints, 3 coords)

来源：

.mat

Audio

1D waveform

来源：

.wav

Precise Temporal Labels

包含：

gesture name
begin frame
end frame

用于：

* frame supervision
* boundary supervision

⸻

4. Data Loading Architecture

4.1 Archive Strategy

数据在：

.tar.gz
    └── SampleXXXX.zip
            ├── *_data.mat
            └── *_audio.wav

因此：

第一阶段

并行解压 tar

第二阶段

并行解析 zip

第三阶段

缓存pickle

⸻

4.2 Parallel IO Design

使用：

ProcessPoolExecutor(max_workers=32)

原因：

* MAT解析CPU-heavy
* WAV读取IO-heavy
* tar解压CPU-heavy

适合多进程。

⸻

5. Feature Engineering Blueprint

⸻

5.1 Motion Features

Skeleton原始维度：

20 × 3 = 60

需要扩展为：

363 dims

推荐包含：

⸻

A. Absolute Joint Position

(x, y, z)

⸻

B. Velocity

Δx, Δy, Δz

⸻

C. Acceleration

Δ²x, Δ²y, Δ²z

⸻

D. Bone Features

骨骼长度：

joint_i - joint_j

⸻

E. Angle Features

关节夹角：

arccos(...)

⸻

F. Relative Coordinates

相对于：

* torso
* shoulder center
* hip center

归一化。

⸻

5.2 Audio Features

输出：

192 dims

推荐：

⸻

A. Log-Mel Spectrogram

核心特征。

⸻

B. MFCC

补充频域信息。

⸻

C. Delta / Delta-Delta

动态变化。

⸻

D. Energy

手势节奏信息。

⸻

5.3 Final Fusion Feature

最终：

363 motion
+
192 audio
=
555 dims

输入主模型。

⸻

6. Validation Blueprint（最关键）

⸻

6.1 为什么不能Random Split

该任务：

User-independent learning

如果随机split：

同一个session
同时出现在train和val

会：

* CV虚高
* leaderboard崩塌

⸻

6.2 正确方案

ShuffledGroupKFold

group：

sample['id']

按session切。

⸻

6.3 Split Logic

unique_groups
    ↓
KFold on groups
    ↓
map back to sample indices

本质：

Group-aware shuffled CV

⸻

7. Model Blueprint

核心模型：

MSTC + BiGRU + Attention

⸻

7.1 Architecture Overview

Motion Features
        ↓
GLU Gating
        ↓
MSTC
        ↓
BiGRU
        ↓
MultiHeadAttention
        ↓
Dual Heads
    ├── Gesture Classification
    └── Boundary Detection

⸻

8. Modality Fusion

⸻

8.1 为什么需要Gating

音频并不总是有效。

Skeleton也不总稳定。

因此：

需要：

Adaptive modality weighting

⸻

8.2 GLU Fusion

对于motion/audio：

Linear
    ↓
GLU

即：

x * sigmoid(gate)

优势：

* 自动选择模态
* 动态抑制噪声
* 比concat稳定

⸻

9. MSTC（核心Temporal CNN）

⸻

9.1 为什么需要多尺度卷积

Gesture长度变化巨大：

短动作
长动作
重复动作

因此：

单kernel不够。

⸻

9.2 Multi-Scale Kernels

并行：

k = 3
k = 5
k = 7
k = 9

提取：

* micro motion
* macro motion

⸻

9.3 MSTC优势

相比单Conv：

* 更强时间感受野
* 更稳定gesture onset
* 更适合边界建模

⸻

10. BiGRU Temporal Modeling

⸻

10.1 为什么不用纯Transformer

该任务：

* 数据量不算大
* 序列长度变化大
* gesture边界很重要

BiGRU：

* 小数据更稳
* 边界更敏感
* temporal continuity更强

⸻

10.2 双向建模

Gesture判断依赖：

过去
+
未来

因此：

BiGRU

优于单向RNN。

⸻

11. Attention Layer

⸻

11.1 Attention作用

解决：

长距离gesture依赖

例如：

gesture A
pause
gesture A continuation

Attention可以：

* 聚合远程信息
* 修复局部误判

⸻

11.2 Residual Attention

使用：

gru_out + attn_out

避免attention破坏原时序结构。

⸻

12. Dual-Head Design

⸻

12.1 主分类头

输出：

21 classes

包括：

0 = background
1-20 = gestures

⸻

12.2 Boundary Head

额外预测：

gesture transition probability

作用：

* 强化边界学习
* 防止gesture merge
* 提高sequence质量

⸻

13. Loss Blueprint

⸻

13.1 Focal Loss

因为：

背景帧远多于gesture帧。

因此：

普通CE会：

偏向background

使用：

Focal Loss

抑制easy negatives。

⸻

13.2 Boundary Weighting

transition附近：

weight = 2.0

原因：

metric主要损失来自：

边界错误

而不是中心区域。

⸻

13.3 Boundary BCE Loss

辅助loss：

BCE(boundary)

提升：

* segmentation precision
* onset detection

⸻

14. Training Blueprint

⸻

14.1 Distributed Training

使用：

DDP

而不是DataParallel。

⸻

14.2 Mixed Precision

torch.cuda.amp

作用：

* 提高吞吐
* 降低显存
* 支持更长序列

⸻

14.3 Optimizer

AdamW

原因：

时序模型比SGD稳定。

⸻

14.4 Scheduler

CosineAnnealingWarmRestarts

作用：

* escape local minima
* stabilize later training

⸻

15. Inference Blueprint

核心：

Frame Probability → Gesture Sequence

⸻

15.1 Logit Averaging

ensemble阶段：

平均 log(prob)

而不是：

平均 prob

原因：

logit averaging：

* 更稳定
* 更保留confidence结构

⸻

15.2 Gaussian Temporal Smoothing

对时间维：

gaussian_filter1d

作用：

* 去除frame jitter
* 平滑边界

⸻

15.3 Background Priority

若：

P(background) > threshold

直接判background。

原因：

防止：

gesture hallucination

⸻

15.4 Duration Filtering

短segment：

len < 15

直接删除。

因为：

大量短gesture其实是噪声。

⸻

15.5 Consecutive Collapse

例如：

2 2 2 5 5 3

转为：

2 5 3

最终得到sequence。

⸻

16. Evaluation Blueprint

metric：

Levenshtein Distance

即：

序列编辑距离。

⸻

16.1 为什么Frame Accuracy没意义

因为：

boundary偏2帧

frame acc几乎不变。

但：

sequence可能完全错误。

⸻

16.2 真正优化目标

优化：

decoded sequence quality

而不是frame classification。

⸻

17. Ensemble Blueprint

⸻

17.1 Fold Ensemble

每fold：

输出：

test_probs

最终：

mean(log probs)

⸻

17.2 为什么OOF重要

OOF：

用于：

* unbiased validation
* ensemble quality estimation

⸻

18. Production Engineering

⸻

18.1 Cache Strategy

缓存：

pickle

避免重复：

* 解压
* mat解析
* wav读取

⸻

18.2 Memory Strategy

不要一次加载全部tensor到GPU。

采用：

sample-wise inference

⸻

18.3 CPU-GPU职责分离

CPU：

* decoding
* smoothing
* data parsing

GPU：

* forward/backward

⸻

19. Gold-Level Improvement Directions

⸻

19.1 Stronger Temporal Models

升级：

Conformer
Mamba
TCN-Transformer Hybrid
RWKV

⸻

19.2 Better Decoding

加入：

Viterbi
HMM
CTC-style decoding

⸻

19.3 Boundary Refinement

加入：

Temporal Action Segmentation losses

例如：

* TCN refinement
* MS-TCN

⸻

19.4 Better Audio Fusion

加入：

Cross-modal attention

替代简单GLU。

⸻

19.5 Self-Supervised Pretraining

对Skeleton：

Masked Motion Modeling

对Audio：

HuBERT / wav2vec embeddings

⸻

20. Final Winning Formula

最终本方案的核心其实是：

Session-aware CV
    +
Strong Temporal Modeling
    +
Boundary-sensitive Loss
    +
Sequence-level Decoding
    +
Temporal Smoothing
    +
Logit Ensemble

不是单纯堆大模型。

真正决定LB的：

解码
边界
CV真实性

而不是参数量。


Anti-Fake Blueprint（反伪实现约束蓝图）

用途：

防止LLM生成“结构正确但语义错误”的伪高级代码。

该蓝图不是描述：

应该做什么

而是：

绝对不能怎么做

核心目标：

强制真实数据流

强制真实时序

强制真实模态

禁止伪 temporal modeling

禁止 metadata cheating

⸻

1. Real Temporal Constraint（真实时序约束）

⸻

强制要求

任何Temporal模型：

GRU
LSTM
Transformer
TCN
MSTC
Attention
Mamba
RWKV

输入必须是：

(B, T, D)

且：

T 必须来自真实frame序列

禁止：

x.unsqueeze(1).repeat(...)

禁止：

repeat static vector across temporal axis

禁止：

pseudo sequence generation

⸻

Temporal Validity Rule

必须满足：

Var(x[:, t, :] - x[:, t-1, :]) > epsilon

即：

相邻时间步必须具有真实变化。

如果：

所有时间步几乎相同

则：

判定为：

Fake Temporal Modeling

⸻

2. Real Modality Constraint（真实模态约束）

⸻

Skeleton

必须：

scipy.io.loadmat

解析：

(T, joints, coords)

禁止：

仅使用metadata替代skeleton

禁止：

archive statistics as motion features

⸻

Audio

必须：

wavfile.read
librosa.load
torchaudio.load

生成：

真实时频特征

例如：

* log-mel
* MFCC
* spectrogram

禁止：

用文件大小模拟audio features

禁止：

用sample id生成audio embedding

⸻

3. Feature Authenticity Constraint（特征真实性约束）

⸻

Feature Source Rule

所有feature必须来源于：

raw sensor data

而不是：

filename
archive size
sample id
csv order
directory structure

⸻

禁止伪特征

明确禁止：

sin(sample_id)
cos(sample_id)
log(sample_id)

禁止：

archive_total_size
archive_file_count

禁止：

feature tiling to reach target dimension

禁止：

random nonlinear expansion

⸻

4. No Fake Dimension Matching（禁止伪维度对齐）

⸻

如果蓝图要求：

363 motion dims
192 audio dims

则：

必须：

真实构造这些维度

而不是：

padding
tiling
rolling
repeating
harmonic synthesis

禁止：

while feature.shape[0] < target_dim:

这种“凑维度”逻辑。

⸻

5. Real Boundary Constraint（真实边界约束）

⸻

Boundary supervision必须来自：

frame-level gesture transitions

即：

labels[t] != labels[t-1]

其中：

labels 必须是 frame labels

禁止：

sequence token transitions

例如：

[2,5,3]

之间的变化。

⸻

6. Sequence Integrity Constraint（序列完整性约束）

⸻

如果任务是：

temporal segmentation

则：

禁止：

将sequence token当作frame labels

必须：

保留frame-level supervision

即：

每个frame对应一个label

而不是：

整个sequence只有几个token

⸻

7. Decoder Authenticity Constraint

⸻

Temporal decoder必须输入：

frame-level probabilities

shape：

(T, C)

禁止：

sequence-position logits pretending to be frame logits

⸻

Gaussian smoothing必须作用于：

真实时间轴

而不是：

伪token轴

⸻

8. No Metadata Learning Constraint

⸻

禁止模型主要依赖：

sample id
archive source
filename pattern
directory structure

进行预测。

⸻

如果：

feature importance中：

metadata features dominate

则：

判定：

Invalid Solution

⸻

9. True Multimodal Fusion Constraint

⸻

Multimodal fusion前：

必须存在：

独立 motion encoder
独立 audio encoder

禁止：

metadata concat pretending to be multimodal

⸻

Fusion输入必须来自：

真实时序embedding

而不是：

fixed tabular vector

⸻

10. Temporal Architecture Sanity Check

⸻

如果模型包含：

GRU
Attention
Transformer
TCN

则必须验证：

⸻

时间变化性

Temporal variance exists

⸻

Boundary变化性

Different frames produce different logits

⸻

时序依赖性

随机打乱时间轴后：

metric必须明显下降。

否则：

判定：

Temporal module ineffective

⸻

11. Anti-Pseudo-Sequence Rule

⸻

禁止：

x.repeat(T, ...)

生成序列。

禁止：

copying static features across time

禁止：

synthetic temporal axis from non-temporal data

⸻

12. Engineering Authenticity Constraint

⸻

CPU/GPU职责：

⸻

CPU

允许：

data parsing
feature extraction
decoding
smoothing

⸻

GPU

必须：

真实 temporal forward/backward

禁止：

GPU training over fake repeated tensors

⸻

13. Anti-Blueprint-Imitation Rule

⸻

禁止：

仅复制术语

例如：

* MSTC
* Attention
* Boundary Head
* Temporal Decoder

但：

没有真实对应的数据流。

⸻

如果：

模块存在
但输入语义不成立

则：

判定：

Cosmetic Architecture

⸻

14. Required Semantic Checks（必须通过的语义检查）

生成代码后必须验证：

⸻

Check 1

Skeleton feature：

是否真实来自：

joint trajectories

⸻

Check 2

Audio feature：

是否真实来自：

waveform DSP

⸻

Check 3

Temporal axis：

是否真实来自：

frame progression

⸻

Check 4

Boundary：

是否真实来自：

frame transition

⸻

Check 5

Temporal models：

是否输入真实变化序列。

⸻

15. Final Enforcement Rule

如果：

模型结构复杂
但数据流不是真实时序

则：

该方案视为：

Invalid Temporal Solution

无论：

* attention多复杂
* transformer多深
* 注释多高级
* blueprint术语多完整

都视为：

伪时序实现。
