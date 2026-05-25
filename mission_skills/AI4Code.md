AI4Code 排序任务 GOLD-STABLE BLUEPRINT

基于 GraphCodeBERT + 双流 Transformer + Rank Regression 的 Notebook Cell Ordering 框架

⸻

1. CORE PHILOSOPHY

这个任务本质：

给定 notebook 中：

* code cells
* markdown cells

预测 markdown cell 在整个 notebook 中的正确相对位置。

⸻

该方案的核心不是：

* seq2seq生成
* pointer network
* pairwise排序
* 图搜索
* 复杂beam search
* token-level cross attention over all cells

而是：

将 Notebook Ordering 转化为：

「连续位置回归问题」

即：

* code cell 的顺序天然已知
* markdown cell 只需要预测插入位置

因此：

核心思想：

给所有 code cells 分配：

p_i\in[0,1]

然后让模型学习 markdown cell 的连续位置：

\hat p_j\in[-0.2,1.2]

最后：

* 对所有 cell 的 position 排序
* 得到 notebook 顺序

这是整个方案最关键的 inductive bias。

⸻

2. OVERALL ARCHITECTURE

整体结构：

Code Cells ------> GraphCodeBERT ------\
                                        \
                                         --> Transformer Encoder
                                        /
Markdown Cells -> GraphCodeBERT -------/
Encoder Output ---> Cross-Attention Decoder
                              |
                              v
                     Markdown Position Regressor
                              |
                              v
                  Continuous Position Prediction

⸻

3. DATA PIPELINE

⸻

3.1 基本数据结构

训练输入：

train/
    xxxx.json

每个 notebook:

{
    "cell_id":{
        "cell_type":"code/markdown",
        "source":"..."
    }
}

同时：

train_orders.csv

提供 GT 顺序。

⸻

4. TOKENIZATION STRATEGY

⸻

MUST USE

GraphCodeBERT tokenizer

优先：

microsoft/graphcodebert-base

原因：

* 同时适合 code + markdown
* 比普通 RoBERTa 更适合 notebook

⸻

5. CELL REPRESENTATION

⸻

CRITICAL RULE

不要使用 token-level notebook transformer。

否则：

(#cells × #tokens)^2

复杂度爆炸。

⸻

正确做法：

cell-level encoding

每个 cell：

tokens -> GraphCodeBERT -> CLS vector

仅保留：

last_hidden_state[:,0,:]

即：

CLS embedding

作为 cell representation。

⸻

6. POSITIONAL ENCODING FOR CODE CELLS

⸻

ONLY CODE CELLS HAVE TRUE POSITIONS

code cell 的原顺序天然正确。

因此：

给 code cells 加：

sin/cos positional encoding

位置：

x_i=\frac{i}{n-1}

范围：

[0,1]

⸻

IMPORTANT

markdown cells：

* 不添加真实position
* 让模型学习插入位置

这是核心。

⸻

7. MODEL ARCHITECTURE

⸻

7.1 CODE ENCODER

code embeddings:

x_code

输入：

Transformer Encoder

目的：

* 学习 code cell 间依赖
* 获取 notebook 语义结构

⸻

推荐：

x-transformers Encoder

配置：

depth = 6
heads = 8
dim = 768

⸻

7.2 MARKDOWN DECODER

markdown embeddings:

x_md

进入：

cross-attention decoder

decoder：

* self attention over markdowns
* cross attention to code cells

即：

Markdown attends to Code

这是整个方案最关键部分之一。

⸻

8. REGRESSION HEAD

⸻

markdown decoder output：

x_md

经过：

Linear -> GLU

输出：

position score

推荐：

nn.Sequential(
    nn.Linear(dim,2),
    nn.GLU()
)

最终：

\hat p=-0.2+1.4\cdot\sigma(z)

即允许：

[-0.2,1.2]

原因：

markdown 可能：

* 在最前
* 在最后

必须允许超出 code 范围。

⸻

9. FINAL ORDER CONSTRUCTION

⸻

构造：

all_positions =
[
    code_positions,
    markdown_pred_positions
]

然后：

argsort()

得到 notebook 顺序。

⸻

10. LOSS DESIGN

⸻

CRITICAL

不要使用：

* CrossEntropy
* Pairwise BCE
* Pointwise ranking
* MSE

这些和 leaderboard metric 不一致。

⸻

11. TARGET METRIC

比赛 metric：

Kendall Tau

即：

\tau=1-\frac{4\times inversions}{n(n-1)}

⸻

12. DIFFERENTIABLE APPROXIMATION

⸻

由于 Kendall Tau 不可导：

使用：

SoftRank + Spearman approximation

核心：

torchsort.soft_rank

⸻

推荐 Loss

Spearman rho inspired

L=\frac{6\sum(r_i-\hat r_i)^2}{n(n^2-1)}

优于：

MSE(position)

因为：

它直接优化排序。

⸻

13. TRAINING STRATEGY

⸻

13.1 Notebook Sampling

NOT ALLOWED:

train on all notebook cells directly

原因：

巨大 notebook 会 OOM。

⸻

正确做法

限制：

MAX_CELLS = 128
MAX_TOKENS = 224

⸻

13.2 Code Cell Sampling

当：

n_code > MAX_CELLS

必须：

* 保留 first code cell
* 保留 last code cell
* 中间随机采样

否则：

notebook global structure 崩坏。

⸻

13.3 Markdown Sampling

markdown 可随机采样：

torch.randperm

⸻

14. DATA AUGMENTATION

⸻

HIGHLY RECOMMENDED

使用：

notebook mutation augmentation

例如：

* markdown shuffle
* paraphrase
* markdown insertion
* markdown deletion

同时：

保留 code structure。

⸻

15. TRAIN / VALID SPLIT

⸻

MUST USE

Group Split by ancestor_id

原因：

同源 notebook 泄漏极其严重。

必须：

GroupShuffleSplit(groups=ancestor_id)

⸻

16. OPTIMIZER

⸻

推荐：

AdamW

学习率：

2e-5 ~ 1e-4

⸻

17. PARAMETER GROUPS

⸻

必须：

GraphCodeBERT lr smaller
Head lr larger

例如：

bert_lr = 2e-5
head_lr = 1e-4

否则：

encoder 容易 catastrophic forgetting。

⸻

18. SCHEDULER

⸻

推荐：

linear warmup

使用：

get_linear_schedule_with_warmup

warmup：

5%

⸻

19. MIXED PRECISION

⸻

必须：

fp16

否则：

训练太慢。

⸻

20. GRADIENT ACCUMULATION

⸻

推荐：

accumulate_grad_batches = 4~16

因为：

GraphCodeBERT 显存极大。

⸻

21. MULTI-GPU

⸻

推荐：

DDP

NOT DP。

⸻

22. VALIDATION METRIC

⸻

验证必须直接计算：

Kendall Tau

不要只看 loss。

很多情况下：

loss下降
但tau不上升

⸻

23. INFERENCE STRATEGY

⸻

推理：

FULL NOTEBOOK

不要随机采样。

必须：

torch.arange(n_cells)

保证 deterministic ordering。

⸻

24. INFERENCE MEMORY OPTIMIZATION

⸻

建议：

batch_size = 1
fp16 inference

避免 OOM。

⸻

25. IMPORTANT IMPLEMENTATION DETAILS

⸻

FORBIDDEN

不允许 token-level global transformer

错误：

all notebook tokens together

复杂度灾难。

⸻

FORBIDDEN

不允许 pairwise ranking explosion

错误：

O(n^2) markdown-code comparisons

⸻

FORBIDDEN

不允许 markdown absolute positional embedding

markdown 没有真实位置。

⸻

FORBIDDEN

不允许 leakage split

绝对不能：

random_split(notebooks)

必须 ancestor grouping。

⸻

26. RECOMMENDED IMPROVEMENTS

⸻

可尝试：

1. Layer dropping

删除 GraphCodeBERT 后几层。

⸻

2. Freeze embeddings

训练更稳定。

⸻

3. Multi-sample inference

多次 markdown sampling 后 ensemble。

⸻

4. Larger decoder depth

decoder 通常比 encoder 更重要。

⸻

5. Auxiliary pairwise loss

可作为 secondary loss。

⸻

27. EXPECTED PERFORMANCE

⸻

该方案特点：

优势

* 极强 inductive bias
* 与 metric 对齐
* 推理简单
* 结构稳定
* 不需要复杂搜索

⸻

劣势

* markdown 极多 notebook 较难
* 长 notebook 仍受限
* GraphCodeBERT 成本高

⸻

28. FINAL GOLD RULES

⸻

THIS TASK IS:

Relative Position Regression

不是：

Sequence Generation

⸻

THE MOST IMPORTANT DESIGN:

固定 code position

只预测：

markdown insertion positions

⸻

THE MOST IMPORTANT LOSS:

SoftRank-based rank correlation optimization

而不是：

MSE

⸻

THE MOST IMPORTANT ARCHITECTURE:

Code Encoder
+
Markdown Cross-Attention Decoder

而不是：

single shared transformer

⸻

RECOMMENDED DEFAULT CONFIG

MAX_CELLS = 128
MAX_TOKENS = 224
encoder_depth = 6
decoder_depth = 6
dim = 768
heads = 8
lr = 2e-5
batch_size = 2
grad_accum = 8
fp16 = True
scheduler = linear_warmup
optimizer = AdamW
backbone = graphcodebert-base

⸻

IDEAL LLM GENERATION TARGET

LLM 生成代码时应满足：

MUST HAVE

* GraphCodeBERT encoder
* Separate code/md streams
* Positional encoding only for code
* Cross-attention markdown decoder
* Continuous regression head
* SoftRank ranking loss
* Kendall Tau validation
* Ancestor-based split
* Cell-level batching
* FP16 support
* DDP compatibility

⸻

FINAL SUMMARY

该方案本质上：

Notebook Structure Modeling
+
Continuous Rank Regression
+
Differentiable Rank Optimization

核心突破：

不是预测“markdown属于哪个code block”，

而是：

直接学习 markdown 在 notebook 中的连续相对位置。