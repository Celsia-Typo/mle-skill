# Blueprint: CHAII — Hindi and Tamil Question Answering

## Overview

This notebook is an **inference-only pipeline** for multilingual extractive question answering in Hindi and Tamil. It loads 5 pretrained XLM-RoBERTa-Large checkpoints (one per fold), runs ensemble inference, and applies language-specific post-processing to extract clean answer spans. The model was fine-tuned on SQuAD v2 and further on the CHAII dataset using a QA span extraction objective.

---

## Competition Details

| Field | Value |
|---|---|
| Competition | [CHAII — Hindi and Tamil Question Answering](https://www.kaggle.com/competitions/chaii-hindi-and-tamil-question-answering) |
| Task | Extractive QA — predict the answer span within a Hindi/Tamil context passage |
| Input | `question` + `context` (in Hindi or Tamil) |
| Target | `PredictionString` — substring of `context` that answers the question |
| Evaluation | Mean word-level Jaccard similarity |
| Reported score | **LB 0.792** |

---

## Dependencies

| Library | Role |
|---|---|
| `torch`, `torch.nn` | Model definition and inference |
| `transformers` | `XLM-RoBERTa`, `AutoTokenizer`, `AutoConfig`, `AdamW`, schedulers |
| `pandas`, `numpy` | Data handling and logit aggregation |
| `apex` (optional) | Mixed-precision training (`fp16`) |
| `collections` | `OrderedDict` for span post-processing |

---

## Configuration (`Config` class)

| Parameter | Value |
|---|---|
| `model_name_or_path` | `xlm-roberta-large-squad-v2` (offline dataset) |
| `max_seq_length` | 400 tokens |
| `doc_stride` | 135 tokens |
| `eval_batch_size` | 128 |
| `learning_rate` | 1e-5 |
| `weight_decay` | 1e-2 |
| `warmup_ratio` | 0.1 |
| `decay_name` | `linear-warmup` |
| `gradient_accumulation_steps` | 2 |
| `fp16` | True if Apex is available |

---

## Pipeline

### 1. Data Loading & Normalization
```python
test = pd.read_csv('test.csv')
test['context']  = test['context'].apply(lambda x: ' '.join(x.split()))
test['question'] = test['question'].apply(lambda x: ' '.join(x.split()))
```
Whitespace normalization removes multi-space gaps common in scraped Hindi/Tamil text.

### 2. Tokenization with Sliding Window (`prepare_test_features`)

Long contexts are handled with sliding window tokenization:
```python
tokenized_example = tokenizer(
    example["question"],
    example["context"],
    truncation="only_second",         # only truncate the context
    max_length=400,
    stride=135,                        # 135-token overlap between windows
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)
```
Each long context produces multiple overlapping feature chunks. `offset_mapping` tracks the character-level span of each token for answer extraction.

### 3. Dataset (`DatasetRetriever`)

Returns per-feature dicts with `input_ids`, `attention_mask`, `offset_mapping`, `sequence_ids`, `context`, `question`, and `example_id` in test mode.

### 4. Model (`Model` — XLM-RoBERTa QA Head)

```python
self.xlm_roberta = AutoModel.from_pretrained(modelname_or_path, config=config)
self.qa_outputs  = nn.Linear(config.hidden_size, 2)   # predicts start and end logits

sequence_output = self.xlm_roberta(input_ids, attention_mask)[0]  # (B, seq_len, hidden)
qa_logits = self.qa_outputs(sequence_output)                       # (B, seq_len, 2)
start_logits, end_logits = qa_logits.split(1, dim=-1)
```

No softmax applied — raw logits are used for ensemble averaging across folds.

### 5. 5-Fold Checkpoint Ensemble

```python
start_logits1, end_logits1 = get_predictions('checkpoint-fold-0/pytorch_model.bin')
# ... repeat for folds 1–4

start_logits = (start_logits1 + start_logits2 + ... + start_logits5) / 5
end_logits   = (end_logits1   + end_logits2   + ... + end_logits5)   / 5
```

Raw logits are averaged across all 5 folds before span decoding. This is more principled than averaging probabilities when logit scales are consistent.

### 6. Span Post-Processing (`postprocess_qa_predictions`)

For each example:
1. Collect all feature chunks (sliding window segments) for this example.
2. For each chunk, enumerate all `(start_index, end_index)` combinations from the top-20 start/end positions.
3. Filter invalid spans: out-of-context tokens, negative length, length > `max_answer_length=30`.
4. Score each valid span: `start_logits[i] + end_logits[j]`.
5. Select the highest-scoring valid span as the predicted answer.
6. Fall back to empty string if no valid spans are found.

### 7. Language-Specific Cleanup

```python
bad_starts   = [".", ",", "(", ")", "-", "–", ",", ";"]
bad_endings  = ["...", "-", "(", ")", "–", ",", ";"]

# Tamil abbreviations that must retain their trailing period:
tamil_ad = "கி.பி"   # AD
tamil_bc = "கி.மு"   # BC
tamil_km = "கி.மீ"   # km

# Hindi abbreviation:
hindi_ad = "ई"
hindi_bc = "ई.पू"
```

Strips leading/trailing punctuation from predicted spans. Special-cases Tamil and Hindi abbreviations ending in `.` that should not be stripped.

---

## Output

| File | Description |
|---|---|
| `submission.csv` | `id` + `PredictionString` (extracted answer span, cleaned) |

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| XLM-RoBERTa-Large | Pretrained on 100 languages including Hindi; superior on Indic scripts vs. mBERT |
| SQuAD v2 initialization | SQuAD v2 trains the model to handle unanswerable questions — important for real QA data |
| `doc_stride=135` with `max_seq_length=400` | Long contexts in Hindi/Tamil Wikipedia passages require sliding window to avoid truncation |
| Raw logit averaging | More stable than probability averaging when model confidence varies across folds |
| n_best_size=20 | Considers top-20 start/end positions, balancing thoroughness with speed |
| Tamil/Hindi abbreviation handling | Domain-specific post-processing required for Indic text punctuation conventions |

---

## Suggested Improvements

- Fine-tune directly on the CHAII training data (this notebook only uses checkpoints from external training).
- Use `IndicBERT` or `MuRIL` which are specifically pretrained on Indic languages.
- Apply character-level normalization for Unicode variations in Devanagari and Tamil scripts.
- Use separate thresholds for start/end logit confidence to handle low-confidence predictions.
- Add null answer detection (CLS token score) to output empty string for unanswerable questions.
