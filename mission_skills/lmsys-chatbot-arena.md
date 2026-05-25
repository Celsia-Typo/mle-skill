# LMSYS Chatbot Arena — Code Generation Prompt

---

## ⚠️ CRASH GUARD — Read This Before Writing Any Code

These are confirmed crash patterns from real runs. Every ❌ below has been observed in traceback logs.

### CG-1 · Deprecated TrainingArguments parameter `evaluation_strategy`
```python
# ❌ CRASHES — removed in transformers ≥ 4.46
TrainingArguments(evaluation_strategy='epoch', ...)

# ✅ ONLY correct form
TrainingArguments(eval_strategy='epoch', ...)
```

### CG-2 · Deprecated TrainingArguments parameter `group_by_length`
```python
# ❌ CRASHES — removed from TrainingArguments in newer transformers
TrainingArguments(group_by_length=True, ...)

# ✅ ONLY correct form — omit this parameter entirely; it no longer exists
TrainingArguments(
    eval_strategy='epoch',
    save_strategy='no',
    # NO group_by_length — does not exist in this transformers version
)
```

### CG-3 · Deprecated `Trainer(tokenizer=...)` argument
```python
# ❌ CRASHES — 'tokenizer' kwarg removed from Trainer.__init__ in transformers ≥ 4.46
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,   # ← causes TypeError immediately
    ...
)

# ✅ ONLY correct form
trainer = Trainer(
    model=model,
    processing_class=tokenizer,   # ← renamed to processing_class
    ...
)
```

### CG-4 · FP16 on H100 with QLoRA → gradient unscaling crash
```python
# ❌ CRASHES at step 0 on H100 — FP16 grad scaler incompatible with QLoRA BF16 compute
BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.float16, ...)
TrainingArguments(fp16=True, ...)

# ✅ ONLY correct form for H100 with QLoRA
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16,   # ← bfloat16, NOT float16
    llm_int8_skip_modules=['score'],
)
# AND in TrainingArguments:
TrainingArguments(
    bf16=True,    # ← bf16, NOT fp16
    # NO fp16=True
    ...
)
# AND when loading base model:
AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,   # ← bfloat16
    ...
)
```
**Why**: H100 natively uses BF16. `fp16=True` activates PyTorch's FP16 GradScaler, which crashes when trying to unscale BF16/4-bit gradients. Error: `ValueError: Attempting to unscale FP16 gradients.`

### CG-5 · NaN in compute_metrics after hours of FP16 training
```python
# ❌ CRASHES after 3-4 hours — FP16 overflow → NaN logits → NaN probs → log_loss fails
# ValueError: Input contains NaN.
def compute_metrics(eval_preds):
    logits = eval_preds.predictions          # can be all-NaN if FP16 overflowed
    probs = softmax(logits)                  # NaN propagates through softmax
    return {'log_loss': log_loss(labels, probs)}   # ← crashes here

# ✅ ONLY correct form — use bf16=True (CG-4 above) AND add NaN guard
def compute_metrics(eval_preds):
    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    if np.any(np.isnan(logits)):
        print("WARNING: NaN logits detected — returning dummy metrics")
        return {'log_loss': 99.0, 'accuracy': 0.0}
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    ll = log_loss(labels, probs)
    preds = np.argmax(logits, axis=-1)
    return {'log_loss': ll, 'accuracy': float((preds == labels).mean())}
```

### CG-6 · Missing `import pandas as pd` or `import gc`
```python
# ❌ CRASHES immediately — NameError: name 'pd' is not defined
train_df = pd.read_csv('./input/train.csv')

# ✅ Ensure these are ALL present at the top of the file:
import gc
import os
import numpy as np
import pandas as pd
import torch
```

### CG-7 · Wrong `remove_unused_columns` setting with custom tokenization
```python
# ❌ CRASHES or silently drops 'label' column — default remove_unused_columns=True
# strips out 'label' when using custom Dataset.map() if 'label' is not in tokenizer output
TrainingArguments(...)  # default remove_unused_columns=True

# ✅ ONLY correct form when mapping with custom tokenize_fn that sets labels manually
TrainingArguments(
    remove_unused_columns=False,   # REQUIRED — keeps 'label' col through Dataset.map
    ...
)
# OR ensure tokenize_fn sets enc['labels'] = examples['label'] explicitly
```

### CG-8 · Time budget overrun — wrong Config values cause 90h+ runtime

**Background**: The H100 80GB is shared across 3 parallel runfiles, so each gets ~1/3 compute.
Real-world timings measured from actual run logs (6.58 s/step observed):

| Config | What the agent generates if you don't specify | Actual runtime | Budget |
|--------|-----------------------------------------------|---------------|--------|
| `max_length=2048, lora_r=64`, both models, full data | ~92h per runfile | 9h | ❌ NEVER FINISHES |
| `max_length=1024, lora_r=16, skip_llama=True, max_train_samples=8000` | ~8-9h per runfile | 9h | ✅ barely fits |

```python
# ❌ SILENT TIME BOMB — agent will generate this from the "ideal" spec; it takes ~92h
class Config:
    max_length        = 2048          # 2048 tokens → 2× slower per step vs 1024
    lora_r            = 64            # rank-64 → ~4× slower than rank-16
    skip_llama        = False         # trains BOTH Gemma-9B AND Llama-8B
    # no max_train_samples            # uses full 57k rows → 14k steps/fold

# ✅ ONLY Config that fits the 9h shared-GPU budget (verified from run logs)
class Config:
    max_length        = 1024          # HARD LIMIT — do NOT increase to 2048
    lora_r            = 16            # HARD LIMIT — do NOT increase to 64
    lora_alpha        = 4
    lora_dropout      = 0.05
    n_splits          = 2
    n_epochs          = 1
    batch_size        = 2
    grad_accum        = 4
    lr                = 2e-4
    warmup_steps      = 20
    use_qlora         = True
    skip_llama        = True          # HARD LIMIT — Llama adds ~40h; skip to stay in budget
    max_train_samples = 8000          # HARD LIMIT — cap training rows; verified ~8h for Gemma 2-fold
    train_100_percent = False
    gemma_weight      = 2
    llama_weight      = 1
    seed              = 42
```

**Why these limits exist** (from observed traceback data):
- `max_train_samples=8000` → ~4312 steps/fold at 6.58 s/step → ~7.9h for 2 Gemma folds ✅
- Full 57k dataset → ~14,350 steps/fold → ~52h for Gemma alone ❌
- `lora_r=64` is ~4× more parameters per layer than `lora_r=16` → each step is significantly slower
- `max_length=2048` doubles sequence length → doubles per-step memory and time
- `skip_llama=True` saves ~40h (Llama-3-8B 2-fold)

> ⚠️ **If using 2-GPU DDP (CG-10)**: training throughput roughly doubles, so the above limits can be relaxed — `lora_r=32`, `max_train_samples=12000`, and `skip_llama=False` become viable within the 9h budget.

**Also required in `main()` — guard the Llama training block:**
```python
# ✅ REQUIRED — without this guard, Llama always trains regardless of skip_llama
if not Config.skip_llama:
    for fold in range(Config.n_splits):
        oof_preds = train_one_fold(fold, train_df, Config,
                                   Config.llama_checkpoint, format_prompt_llama)
        ...
```

**Also required in `train_one_fold()` — subsample training data:**
```python
# ✅ REQUIRED — truncate training set to max_train_samples (stratified)
if hasattr(config, 'max_train_samples') and config.max_train_samples and len(tr_df) > config.max_train_samples:
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=config.max_train_samples, random_state=config.seed)
    idx, _ = next(sss.split(tr_df, tr_df['label']))
    tr_df = tr_df.iloc[idx].reset_index(drop=True)
```

---

**Summary table:**

| Bug | Error message | Crashes when | Fix |
|-----|--------------|-------------|-----|
| CG-1 | `TypeError: unexpected kwarg 'evaluation_strategy'` | Startup (~2 min) | `eval_strategy='epoch'` |
| CG-2 | `TypeError: unexpected kwarg 'group_by_length'` | Startup (~2 min) | Remove entirely |
| CG-3 | `TypeError: unexpected kwarg 'tokenizer'` | Startup (~2 min) | `processing_class=tokenizer` |
| CG-4 | `ValueError: Attempting to unscale FP16 gradients` | Step 0 (~8 min) | `bf16=True` + `bnb_4bit_compute_dtype=bfloat16` |
| CG-5 | `ValueError: Input contains NaN` | After 3-4h | `bf16=True` + NaN guard in compute_metrics |
| CG-6 | `NameError: name 'pd' is not defined` | Startup | `import pandas as pd` |
| CG-7 | Silent label drop or KeyError | Training | `remove_unused_columns=False` |
| CG-8 | Silent timeout — zero output after 9h | Never finishes | See Config above: `max_length=1024`, `lora_r=16`, `skip_llama=True`, `max_train_samples=8000` |
| CG-10 | Training uses only 1 GPU even when 2 are visible | All of training | Add `setup_ddp()` + `ddp_find_unused_parameters=False`; launch with `torchrun --nproc_per_node=2` |

---

## Part 1 · Instructions

You are an expert machine learning engineer. Your goal is to write a **single-file Python script** that solves the LMSYS Chatbot Arena Kaggle competition.

**Task description:**
- Input: CSV with columns `prompt`, `response_a`, `response_b` (each stored as a **string-formatted Python list** — you must parse them with `eval()`).
- Output: probabilities for 3 classes — `winner_model_a`, `winner_model_b`, `winner_tie`.
- Evaluation metric: **log_loss** (lower is better).
- Best strategy: Train **two models** and ensemble their predictions at 2:1 ratio:
  1. `google/gemma-2-9b-it` — QLoRA (rank=64), `n_splits=2`, 1 epoch, `adamw_8bit`, lr=2e-4, max_length=2048. **No TTA.**
  2. `meta-llama/Meta-Llama-3-8B-Instruct` — QLoRA (rank=64), `n_splits=2`, 1 epoch, `adamw_8bit`, lr=2e-4, max_length=2048. **Apply TTA** (swap response_a ↔ response_b, cross-invert logits).
  - Final prediction: `(gemma_probs * 2 + llama_probs_tta * 1) / 3`

**Available input files** (no external datasets):
- `./input/train.csv` — competition training data
- `./input/test.csv` — competition test data
- `./input/sample_submission.csv` — submission format reference

**Hardware & time constraints (CRITICAL):**
- Hardware: **1x NVIDIA H100 80GB**. This GPU is shared across 3 parallel runfiles, giving each runfile **~26GB effective GPU memory** and **~8–9 hours effective wall time**.
- ⚠️ **The time estimate below assumes the REDUCED Config (CG-8). Using the "ideal" Config (max_length=2048, lora_r=64, full data) takes ~92h — it will NEVER finish.**
- Verified time breakdown with `max_length=1024, lora_r=16, max_train_samples=8000, skip_llama=True`:
  - Gemma-2-9B, 2-fold, 8k samples/fold → **~7.9h total** (observed: 6.58 s/step × 4312 steps × 2 folds)
  - Llama-3-8B training: **SKIP** (`skip_llama=True`) — adds ~40h which blows the budget
- `n_splits=2` is mandatory. `n_splits=5` would take 2.5× longer → way over budget.
- `train_100_percent=False` — standard hold-out split preserves a meaningful CV score.

**⚠️ CRITICAL OVERRIDES — READ BEFORE ANY OTHER INSTRUCTION:**

1. **`n_epochs=1` IS CORRECT AND MANDATORY for this task.** Any system instruction saying "use 5-20 epochs" applies to small models (DeBERTa, BERT, etc.) and does NOT apply here. QLoRA fine-tuning of 9B LLMs (Gemma-2-9B, Llama-3-8B) overfits severely beyond 1 epoch on 51k rows. Using more than 1 epoch will increase log_loss. Do not override this.

2. **DO NOT use DeBERTa, ModernBERT, RoBERTa, or any model smaller than 7B parameters.** These produce log_loss ~0.95–1.0, which is below the medal threshold. This task REQUIRES 9B-scale LLMs via QLoRA to be competitive.

3. **Both gated models ARE accessible in this environment.** `google/gemma-2-9b-it` and `meta-llama/Meta-Llama-3-8B-Instruct` can be downloaded. The HF_TOKEN is available at runtime — do not replace these models with ungated alternatives.

**Constraints:**
- The script must be self-contained and runnable end-to-end.
- **Gated models require HF_TOKEN.** Both `google/gemma-2-9b-it` and `meta-llama/Meta-Llama-3-8B-Instruct` are gated. Set this at the very top before any `from_pretrained()` call:
  ```python
  import os
  os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', 'hf_placeholder')  # injected at runtime
  os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
  import transformers.modeling_utils
  transformers.modeling_utils.check_torch_load_is_safe = lambda: None
  ```
- Before writing the final code, **first sketch the overall problem-solving process** using the three programming structures:
  - **Sequential structure**: list the main pipeline steps in order.
  - **Branch structure**: identify key conditional decisions (TTA for Llama only, model-specific tokenizer/template).
  - **Loop structure**: identify key iteration points (n_splits=2 folds × 2 models, batch loop).

Then output the complete Python script.

---

## Part 2 · Demonstration Examples

### Example 1

**Requirement:**
```python
def parse_field(val) -> str:
    """Parse a string-formatted Python list column into a single joined string."""
```

**SCoT:**
```
# Input:  val: any  (raw CSV cell value — may be str, float/NaN, or None)
# Output: text: str (joined plain text)

# Sequential:
#   step 1 → if val is None or float(NaN): return ''
#   step 2 → parsed = eval(str(val), {'null': ''})
#   step 3 → if parsed is list: return ' '.join([str(x) for x in parsed if x is not None])
#   step 4 → else: return str(parsed)

# Branch:
#   if val is None or isinstance(val, float) → return ''
#   elif parsed is list                       → join elements
#   else                                      → cast to str

# Loop:
#   for x in parsed: skip None, cast to str, join
```

**Code:**
```python
def parse_field(val: object) -> str:
    """Parse a string-formatted Python list column into a single joined string."""
    # step 1: handle missing / NaN values
    if val is None or isinstance(val, float):
        return ''
    try:
        # step 2: safely eval the string representation
        parsed = eval(str(val), {'null': ''})
        # step 3: join list elements, skip None
        if isinstance(parsed, list):
            return ' '.join([str(x) for x in parsed if x is not None])
        # step 4: fallback cast
        return str(parsed)
    except Exception:
        return str(val)
```

---

### Example 2

**Requirement:**
```python
def build_model(checkpoint: str, lora_config, use_qlora: bool) -> nn.Module:
    """Load a HuggingFace model with optional 4-bit QLoRA and LoRA adapters for 3-class classification."""
```

**SCoT:**
```
# Input:  checkpoint: str      (HuggingFace model id)
#         lora_config: LoraConfig
#         use_qlora:  bool
# Output: model: nn.Module     (PEFT-wrapped AutoModelForSequenceClassification)

# Sequential:
#   step 1 → if use_qlora: build BitsAndBytesConfig (4-bit nf4, compute_dtype=float16)
#   step 2 → load AutoModelForSequenceClassification (num_labels=3, torch_dtype=float16)
#   step 3 → if use_qlora: prepare_model_for_kbit_training(model)
#   step 4 → model = get_peft_model(model, lora_config)
#   step 5 → return model

# Branch:
#   if use_qlora → add quantization_config + call prepare_model_for_kbit_training
#   else         → load in full fp16, skip kbit prep

# Loop: (none)
```

**Code:**
```python
def build_model(checkpoint: str, lora_config, use_qlora: bool):
    """Load model with optional 4-bit QLoRA and LoRA adapters."""
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    # NOTE: correct LoraConfig parameter names
    # LoraConfig(
    #     r=64,
    #     lora_alpha=4,
    #     lora_dropout=0.05,   # ← 'lora_dropout', NOT 'dropout'
    #     target_modules=[...],
    #     bias="none",
    #     task_type=TaskType.SEQ_CLS,
    #     modules_to_save=["score"],
    # )

    # NOTE: use bfloat16, NOT float16 — H100 uses BF16 natively; fp16 causes gradient unscaling crash (CG-4)
    kwargs = dict(num_labels=3, torch_dtype=torch.bfloat16, device_map='auto',
                  trust_remote_code=True)

    # step 1: configure 4-bit quantization
    if use_qlora:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,  # ← bfloat16, NOT float16 (CG-4)
            llm_int8_skip_modules=['score'],
        )

    # step 2: load base model
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, **kwargs)

    # step 3: kbit prep
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    # step 4: inject LoRA adapters
    model = get_peft_model(model, lora_config)
    return model
```

---

### Example 3

**Requirement:**
```python
def ensemble_predictions(gemma_probs: np.ndarray, llama_probs: np.ndarray,
                         weights: tuple = (2, 1)) -> np.ndarray:
    """Weighted average of Gemma-2 and Llama-3 fold-averaged probabilities at 2:1 ratio."""
```

**SCoT:**
```
# Input:  gemma_probs: np.ndarray  shape (N, 3)
#         llama_probs: np.ndarray  shape (N, 3) — TTA already applied
#         weights: tuple           default (2, 1)
# Output: final_probs: np.ndarray  shape (N, 3)

# Sequential:
#   step 1 → w_total = sum(weights)
#   step 2 → final = (gemma_probs * weights[0] + llama_probs * weights[1]) / w_total
#   step 3 → clip to [0, 1]
#   step 4 → return final

# Branch/Loop: none — numpy broadcasting handles everything
```

**Code:**
```python
def ensemble_predictions(gemma_probs: np.ndarray, llama_probs: np.ndarray,
                         weights: tuple = (2, 1)) -> np.ndarray:
    """Weighted average at 2:1 ratio."""
    w_total = sum(weights)
    final = (gemma_probs * weights[0] + llama_probs * weights[1]) / w_total
    return np.clip(final, 0.0, 1.0)
```

---

## Part 3 · Testing Requirement

Now implement the **complete single-file Python script**. Fill in each function body according to the SCoT comments.

```python
# ============================================================
# BOILERPLATE (MUST be first — do NOT move)
# ============================================================
import os
os.environ['HF_TOKEN'] = 'YOUR_TOKEN'      # REQUIRED: gated model access
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

import ast
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EvalPrediction, BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from datasets import Dataset

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    gemma_checkpoint  = 'google/gemma-2-9b-it'
    llama_checkpoint  = 'meta-llama/Meta-Llama-3-8B-Instruct'
    max_length        = 1024   # ⚠️ CG-8: HARD LIMIT — 2048 doubles runtime, budget overflow
    n_splits          = 2      # MUST be 2 — n_splits=5 exceeds 8h time budget on shared H100
    n_epochs          = 1      # MANDATORY — QLoRA 9B overfits beyond 1 epoch
    batch_size        = 2
    grad_accum        = 4
    lr                = 2e-4
    warmup_steps      = 20
    lora_r            = 16     # ⚠️ CG-8: HARD LIMIT — rank=64 is 4× slower, budget overflow
    lora_alpha        = 4
    lora_dropout      = 0.05   # NOTE: 'lora_dropout', NOT 'dropout'
    use_qlora         = True   # ~12-15GB VRAM per model, fits within 26GB budget
    skip_llama        = True   # ⚠️ CG-8: HARD LIMIT — Llama adds ~40h; always skip
    max_train_samples = 8000   # ⚠️ CG-8: HARD LIMIT — verified: 8k rows → ~7.9h for Gemma 2-fold
    train_100_percent = False  # False saves ~20% time and gives meaningful CV
    gemma_weight      = 2
    llama_weight      = 1
    seed              = 42

# ============================================================
# DATA PIPELINE
# ============================================================
def parse_field(val) -> str:
    """
    # Input:  val: any  (raw CSV cell — may be str, float/NaN, None)
    # Output: text: str
    # Branch: if NaN/None → ''  |  if list → join  |  else → str cast
    """
    pass

def sanitize_text(text: str) -> str:
    """Strip invalid UTF-8 bytes to prevent PyArrow crashes during Dataset.from_pandas()."""
    return text.encode('utf-8', 'ignore').decode('utf-8')

def load_data(train_path: str, test_path: str):
    """
    # Input:  train_path: str, test_path: str
    # Output: train_df, test_df: pd.DataFrame
    # Sequential: read csvs → parse+sanitize text cols → derive int label → return
    # Loop:    for col in ['prompt','response_a','response_b']: apply parse_field + sanitize
    """
    pass

def create_folds(df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """
    # Input:  df: DataFrame with 'label' column, n_splits: int (use 2)
    # Output: df with 'fold' column  (StratifiedKFold, random_state=Config.seed)
    # Sequential: init StratifiedKFold → enumerate splits → assign fold id → return
    """
    pass

def format_prompt_gemma(row, swap=False) -> str:
    """
    # IMPORTANT: swap=False parameter MUST be present even though Gemma never uses it.
    # inference_one_model() calls format_fn(row, swap=swap) for ALL models.
    # Missing swap=False causes: TypeError: unexpected keyword argument 'swap'
    #
    # Input:  row: dict with 'prompt','response_a','response_b', swap: bool (ignored)
    # Output: formatted string using Gemma <start_of_turn> template
    # Template: <start_of_turn>prompt\n{p}<end_of_turn>\n
    #           <start_of_turn>response_a\n{a}<end_of_turn>\n
    #           <start_of_turn>response_b\n{b}<end_of_turn>
    """
    pass

def format_prompt_llama(row, swap=False) -> str:
    """
    # Input:  row: dict, swap: bool (if True, swap response_a and response_b for TTA)
    # Output: formatted string using [PROMPT]/[RESPONSE_A]/[RESPONSE_B] template
    # Branch:  if swap → exchange a and b before formatting
    """
    pass

# ============================================================
# MODEL  (shared builder for Gemma-2 and Llama-3)
# ============================================================
def build_model(checkpoint: str, lora_config: LoraConfig, use_qlora: bool, local_rank: int = -1):
    """
    # Input:  checkpoint, lora_config, use_qlora, local_rank (-1 = single-GPU, ≥0 = DDP rank)
    # Output: PEFT-wrapped AutoModelForSequenceClassification
    # Sequential: bnb_config → from_pretrained → kbit_prep → get_peft_model
    # Branch:  use_qlora → add quantization_config + prepare_model_for_kbit_training
    #          local_rank >= 0 → device_map={'': local_rank}  else → device_map='auto'  (CG-10)
    # WARNING: LoraConfig parameter is 'lora_dropout', NOT 'dropout'
    """
    pass

# ============================================================
# TRAINING
# ============================================================
def compute_metrics(eval_preds: EvalPrediction) -> dict:
    """
    # Input:  logits ndarray, label_ids int ndarray
    # Output: {'log_loss': float, 'accuracy': float}
    """
    pass

def train_one_fold(fold_idx: int, train_df: pd.DataFrame, config: Config,
                   checkpoint: str, format_fn) -> np.ndarray:
    """
    # Input:  fold_idx, train_df (with 'fold' col), config, checkpoint, format_fn
    # Output: oof_probs: np.ndarray (N_val, 3)
    # Sequential:
    #   step 1 → split: tr_df = fold != fold_idx, val_df = fold == fold_idx
    #   step 2 → tokenizer = AutoTokenizer(checkpoint); set pad_token if None
    #   step 3 → build Dataset, map format_fn+tokenize (batched=True)
    #   step 4 → build LoraConfig(lora_dropout=config.lora_dropout) → build_model
    #   step 5 → TrainingArguments(bf16=True, optim='adamw_8bit') + Trainer  ← bf16 NOT fp16 (CG-4)
    #   step 6 → trainer.train() → predict(val_ds) → softmax → save adapter → cleanup
    # Branch:  if train_100_percent → tr_df = entire train_df (not used here, False)
    # Loop:    Trainer handles batching internally
    # WARNING: save adapter with model.save_pretrained(path) — saves LoRA weights only
    # WARNING: Trainer must use processing_class=tokenizer NOT tokenizer=tokenizer (CG-3)
    # WARNING: TrainingArguments must use eval_strategy NOT evaluation_strategy (CG-1)
    # WARNING: do NOT add group_by_length=True — removed from TrainingArguments (CG-2)
    # WARNING: add remove_unused_columns=False to prevent label column being dropped (CG-7)
    """
    pass

# ============================================================
# INFERENCE
# ============================================================
def inference_one_model(test_df: pd.DataFrame, model_paths: list,
                        config: Config, checkpoint: str,
                        format_fn, apply_tta: bool) -> np.ndarray:
    """
    # Input:  test_df, model_paths (adapter dirs), checkpoint, format_fn, apply_tta
    # Output: avg_probs: np.ndarray (N_test, 3)
    # Sequential:
    #   step 1 → tokenizer = AutoTokenizer(checkpoint)
    #   step 2 → for each adapter path:
    #              CORRECT loading pattern:
    #                base = AutoModelForSequenceClassification.from_pretrained(
    #                    checkpoint, num_labels=3, torch_dtype=float16,
    #                    device_map='auto', quantization_config=BitsAndBytesConfig(...)
    #                )
    #                model = PeftModel.from_pretrained(base, path)
    #              WRONG: AutoModelForSequenceClassification.from_pretrained(path)
    #                     ← adapter dirs don't contain full weights, will fail
    #   step 3 → predict original order → logits_orig
    #   step 4 → if apply_tta: predict swap=True → logits_tta
    #              cross-invert: combined[:,0] = orig[:,0]*0.5 + tta[:,1]*0.5
    #                            combined[:,1] = orig[:,1]*0.5 + tta[:,0]*0.5
    #                            combined[:,2] = orig[:,2]*0.5 + tta[:,2]*0.5
    #   step 5 → softmax → accumulate → average over folds → return
    # Branch:  apply_tta=False for Gemma, apply_tta=True for Llama
    # Loop:    for path in model_paths
    """
    pass

def ensemble_predictions(gemma_probs: np.ndarray, llama_probs: np.ndarray,
                         weights: tuple = (2, 1)) -> np.ndarray:
    """Weighted average: (gemma*2 + llama*1) / 3, clipped to [0,1]."""
    pass

# ============================================================
# MAIN
# ============================================================
def main():
    local_rank = setup_ddp()   # CG-10: MUST be first — returns -1 if not torchrun
    os.makedirs('./working', exist_ok=True)
    os.makedirs('./submission', exist_ok=True)

    # 1. Load  [Sequential]
    train_df, test_df = load_data('./input/train.csv', './input/test.csv')

    # 2. Folds  [Sequential]
    train_df = create_folds(train_df, Config.n_splits)

    # 3. Train Gemma (n_splits=2, no TTA)  [Loop]
    oof_gemma, gemma_paths = np.zeros((len(train_df), 3)), []
    for fold in range(Config.n_splits):
        print(f'[Gemma] Fold {fold+1}/{Config.n_splits}')
        oof_preds = train_one_fold(fold, train_df, Config,
                                   Config.gemma_checkpoint, format_prompt_gemma)
        oof_gemma[train_df[train_df['fold'] == fold].index] = oof_preds
        gemma_paths.append(f'./working/gemma_fold{fold}')

    # 4. Train Llama (n_splits=2, TTA at inference)  [Loop]
    oof_llama, llama_paths = np.zeros((len(train_df), 3)), []
    for fold in range(Config.n_splits):
        print(f'[Llama] Fold {fold+1}/{Config.n_splits}')
        oof_preds = train_one_fold(fold, train_df, Config,
                                   Config.llama_checkpoint, format_prompt_llama)
        oof_llama[train_df[train_df['fold'] == fold].index] = oof_preds
        llama_paths.append(f'./working/llama_fold{fold}')

    # 5. CV  [Sequential]
    labels = train_df['label']
    print(f'Gemma  CV: {log_loss(labels, oof_gemma):.4f}')
    print(f'Llama  CV: {log_loss(labels, oof_llama):.4f}')
    oof_ens = ensemble_predictions(oof_gemma, oof_llama,
                                   (Config.gemma_weight, Config.llama_weight))
    print(f'Ensemble CV: {log_loss(labels, oof_ens):.4f}')

    # 6. Test inference  [Sequential]
    gemma_test = inference_one_model(test_df, gemma_paths, Config,
                                     Config.gemma_checkpoint, format_prompt_gemma,
                                     apply_tta=False)
    llama_test = inference_one_model(test_df, llama_paths, Config,
                                     Config.llama_checkpoint, format_prompt_llama,
                                     apply_tta=True)

    # 7. Ensemble + submit  [Sequential]
    final = ensemble_predictions(gemma_test, llama_test,
                                 (Config.gemma_weight, Config.llama_weight))
    sub = pd.DataFrame(final, columns=['winner_model_a', 'winner_model_b', 'winner_tie'])
    sub.insert(0, 'id', test_df['id'])
    sub.to_csv('./submission/submission.csv', index=False)
    print(f'Done. Rows: {len(sub)}')
    cleanup_ddp()              # CG-10: MUST be last

if __name__ == '__main__':
    main()
```

---

### CG-9 · Do NOT hardcode CUDA_VISIBLE_DEVICES inside the script

# ❌ WRONG — overrides the GPU assignment injected by the scheduler
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # ← never write this in generated code

# ✅ CORRECT — CUDA_VISIBLE_DEVICES is already set before your code runs
# Just use device_map='auto' or device='cuda' and the correct GPU is used automatically
model = AutoModel.from_pretrained(checkpoint, device_map='auto', ...)
device = 'cuda'   # NOT 'cuda:0' — CUDA_VISIBLE_DEVICES already remapped the index
---

### CG-10 · Single-GPU training wastes resources when 2 GPUs are available — implement DDP

**Background**: `CUDA_VISIBLE_DEVICES` is pre-set by the scheduler per runfile. When it exposes 2 GPUs, `device_map='auto'` only **shards model weights** across them — it does NOT parallelise the training data. The `Trainer` without DDP still trains on a single GPU. To actually utilise both GPUs, the script must be launched with `torchrun` and include DDP initialisation.

> ❌ **Do NOT use `nn.DataParallel`** with QLoRA. `DataParallel` replicates the PEFT adapter across GPU threads; bitsandbytes 4-bit layers are not thread-safe in this mode → silent NaN gradients or `RuntimeError` during backward.

**Step 1 — add DDP helpers after all imports, before `class Config`:**
```python
import torch.distributed as dist

def setup_ddp() -> int:
    """Init NCCL process group when launched via torchrun; return local_rank (-1 if single-GPU)."""
    if 'LOCAL_RANK' not in os.environ:
        return -1
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return local_rank

def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
```

**Step 2 — update `build_model()` signature:**
```python
# ❌ Hard-codes device placement; DDP ranks collide on the same GPU
def build_model(checkpoint, lora_config, use_qlora):
    kwargs = dict(..., device_map='auto', ...)

# ✅ Pin each DDP rank to its own GPU; fall back to 'auto' for single-GPU runs
def build_model(checkpoint: str, lora_config: LoraConfig, use_qlora: bool, local_rank: int = -1):
    device_map = {'': local_rank} if local_rank >= 0 else 'auto'
    kwargs = dict(num_labels=3, torch_dtype=torch.bfloat16,
                  device_map=device_map, trust_remote_code=True)
    ...
```

**Step 3 — add `ddp_find_unused_parameters=False` to `TrainingArguments` in `train_one_fold()`:**
```python
# ❌ Missing flag → DDP hangs or crashes on PEFT layers with unused parameters
TrainingArguments(bf16=True, optim='adamw_8bit', ...)

# ✅ REQUIRED for DDP + PEFT — without this, DDP times out waiting for unused adapter params
TrainingArguments(
    bf16=True,
    optim='adamw_8bit',
    ddp_find_unused_parameters=False,   # ← REQUIRED when using DDP with LoRA/PEFT
    ...
)
```

**Step 4 — update `main()` entry and exit:**
```python
# ❌ No DDP initialisation — Trainer silently uses only rank-0 GPU
def main():
    os.makedirs('./working', exist_ok=True)
    ...

# ✅ REQUIRED — setup_ddp() MUST be the very first call in main()
def main():
    local_rank = setup_ddp()   # ← first line; returns -1 if not launched via torchrun
    os.makedirs('./working', exist_ok=True)
    os.makedirs('./submission', exist_ok=True)
    ...
    # Pass local_rank to every build_model() call:
    model = build_model(checkpoint, lora_config, config.use_qlora, local_rank=local_rank)
    ...
    cleanup_ddp()              # ← last line of main()
```

**Launch command:**
```bash
# ✅ Multi-GPU (e.g. 2× H100) — linear training speedup
torchrun --nproc_per_node=2 runfile_0.py

# Single-GPU fallback — identical behaviour; LOCAL_RANK not set → setup_ddp() returns -1
python runfile_0.py
```

**With 2-GPU DDP the CG-8 hard limits can be relaxed** (estimated ~4h for Gemma 2-fold):
```python
# ✅ Relaxed Config for 2-GPU DDP runs (single-GPU limits in comments)
class Config:
    max_length        = 1024   # keep 1024 — VRAM limit per GPU unchanged
    lora_r            = 32     # relaxed from 16 → 32 (2× speedup absorbs the cost)
    lora_alpha        = 4
    lora_dropout      = 0.05
    n_splits          = 2
    n_epochs          = 1
    batch_size        = 2
    grad_accum        = 4
    lr                = 2e-4
    warmup_steps      = 20
    use_qlora         = True
    skip_llama        = False  # relaxed from True → can train Llama (~4h extra with 2 GPUs)
    max_train_samples = 12000  # relaxed from 8000 → 12000
    train_100_percent = False
    gemma_weight      = 2
    llama_weight      = 1
    seed              = 42
```

> ⚠️ If `skip_llama=False` with DDP, total runtime ~8h (both models). Still within 9h budget — but only if `ddp_find_unused_parameters=False` is present and `local_rank` is correctly threaded through.

**Summary of required DDP changes:**

| Location | What to change |
|----------|---------------|
| After imports, before `class Config` | Add `setup_ddp()` + `cleanup_ddp()` |
| `build_model()` signature | Add `local_rank: int = -1`; `device_map={'': local_rank}` when `local_rank >= 0` |
| `train_one_fold()` → `TrainingArguments` | Add `ddp_find_unused_parameters=False` |
| `main()` first line | `local_rank = setup_ddp()` |
| `main()` last line | `cleanup_ddp()` |
| Launch command | `torchrun --nproc_per_node=2 script.py` |

---

## Part 4 · Trigger Prompts

Let's think step by step.

Before writing any function body, first write a brief outline using sequential / branch / loop labels.

Write your complete code here.
