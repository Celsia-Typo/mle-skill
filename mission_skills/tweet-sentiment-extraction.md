GOLD-STABLE BLUEPRINT — Tweet Sentiment Extraction (DeBERTa-v3-Large QA Ensemble)

1. CORE PHILOSOPHY

This competition is NOT solved by:

* vanilla classification
* seq2seq generation
* naive token classification
* excessive preprocessing
* complex postprocessing
* aggressive text normalization
* public LB probing
* huge epoch counts

This competition is solved by:

* extractive QA framing
* precise character-token alignment
* stable span supervision
* strong transformer backbone
* fold-level logit ensembling
* neutral-text heuristics
* robust decoding constraints
* inference consistency

The objective is maximizing:

* character-level span fidelity
* Jaccard similarity
* decoding stability
* cross-fold consistency

⸻

2. SINGLE-FILE REQUIREMENTS

The generated .py solution MUST:

* contain the entire pipeline in one file
* run end-to-end from raw CSVs
* support full training + inference
* support 5-fold CV
* support multi-GPU DDP
* generate submission.csv
* avoid notebook-only code
* avoid hidden dependencies
* avoid interactive steps

Pipeline order MUST be:

1. load_data
2. splitter creation
3. preprocessing
4. fold training
5. OOF/test prediction
6. fold ensembling
7. span decoding
8. submission generation

⸻

3. DATA LOADING RULES

Use:

* pd.read_csv(..., keep_default_na=False, dtype=str)

This is MANDATORY.

Reason:

* whitespace integrity matters
* substring alignment matters
* NaN conversion corrupts spans
* empty-string corruption destroys Jaccard

NEVER:

* strip whitespace globally
* lowercase text globally
* normalize punctuation
* collapse spaces

Required columns:

Train:

* text
* sentiment
* selected_text

Test:

* text
* sentiment
* textID

⸻

4. VALIDATION STRATEGY

Use:

* 5-fold StratifiedKFold
* stratify on sentiment only
* shuffle=True
* random_state=42

Rationale:

Sentiment distribution is imbalanced and strongly affects:

* span lengths
* neutral identity mapping
* decoding behavior

DO NOT:

* use plain KFold
* stratify on text length
* stratify on selected_text
* use GroupKFold

⸻

5. MODELING PHILOSOPHY

Frame the task as:

Extractive Question Answering

Input format MUST be:

[CLS] sentiment [SEP] text [SEP]

Target:

* predict start token
* predict end token

DO NOT:

* generate text autoregressively
* use classification labels
* predict BIO tags
* use seq2seq decoding

⸻

6. TOKENIZER REQUIREMENTS

Use:

AutoTokenizer.from_pretrained(
    "microsoft/deberta-v3-large",
    use_fast=True
)

Fast tokenizer is MANDATORY because:

* offset mapping is required
* precise span alignment is required

Required tokenizer settings:

max_length=160
padding='max_length'
truncation=True
return_offsets_mapping=True

⸻

7. SPAN ALIGNMENT RULES

This is the MOST CRITICAL PART of the solution.

The generated solution MUST:

A. Use character-level alignment

Convert:

* selected_text character spans
    → token spans

using:

* offset_mapping
* sequence_ids()

⸻

B. Restrict supervision to TEXT TOKENS ONLY

Use:

sequence_id == 1

Never allow:

* sentiment tokens
* special tokens
* padding tokens

to become valid answer spans.

⸻

C. Multi-Occurrence Handling

If selected_text appears multiple times:

DO NOT:

* blindly use first occurrence

Instead:

* locate all matches
* choose the occurrence whose midpoint is closest to the text midpoint

This stabilizes:

* supervision
* decoding
* Jaccard consistency

⸻

8. MODEL ARCHITECTURE

Use:

DeBERTa-v3-Large

Required properties:

* hidden size 1024
* QA-style span head
* start/end logits

Head structure:

Linear(1024 -> 2)

Output:

* start_logits
* end_logits

⸻

9. MULTI-SAMPLE DROPOUT

MUST implement:

Vectorized Multi-Sample Dropout

Recommended:

* 5 dropout samples
* dropout rate 0.5

Workflow:

1. apply multiple dropout masks
2. compute logits for each
3. average logits

This improves:

* generalization
* fold stability
* ensemble quality

DO NOT:

* use Python-side detached averaging
* break autograd graph
* use non-vectorized loops that destabilize DDP

⸻

10. TRAINING CONFIGURATION

Recommended:

epochs = 3
lr = 1e-5
weight_decay = 0.01
batch_size_per_gpu = 8

Scheduler:

* linear warmup
* warmup ratio 0.1

Loss:

* CrossEntropyLoss
* label_smoothing=0.1

⸻

11. DDP REQUIREMENTS

The generated solution MUST support:

DistributedDataParallel

Required:

static_graph=True

Enable:

gradient_checkpointing_enable()

Purpose:

* memory reduction
* synchronization stability
* avoid double-ready errors

DO NOT:

* use DataParallel
* use manual gradient sync
* disable DistributedSampler

⸻

12. CHECKPOINT AVERAGING

MANDATORY.

Save:

* last 2 epochs

Then compute:

arithmetic weight average

This acts as lightweight SWA.

Benefits:

* smoother minima
* stabler logits
* stronger LB consistency

DO NOT:

* average optimizer states
* average random checkpoints
* average early unstable epochs

⸻

13. INFERENCE RULES

Inference MUST output:

start_logits
end_logits

NOT decoded strings.

Reason:

* fold ensemble must occur in logit space

⸻

14. ENSEMBLE STRATEGY

MANDATORY:

Logit Averaging

Average:

* start logits
* end logits

across folds.

DO NOT:

* average decoded strings
* majority vote substrings
* average token indices

Correct ensemble space:

* logits only

⸻

15. DECODING RULES

For non-neutral samples:

Search for:

argmax(start_logit[s] + end_logit[e])

subject to:

e >= s

AND:

* both tokens belong to text segment

This constraint is CRITICAL.

DO NOT:

* decode outside text tokens
* allow invalid spans
* independently argmax start/end

⸻

16. HEURISTIC OVERRIDES

These heuristics are HIGH VALUE.

A. Neutral Override

If:

sentiment == "neutral"

Return:

full text

This is near-optimal.

⸻

B. Short Text Override

If:

len(text.split()) <= 3

Return:

full text

This significantly improves:

* short tweet stability
* punctuation handling
* noisy spans

DO NOT:

* attempt aggressive extraction on ultra-short tweets

⸻

17. OUTPUT RECONSTRUCTION

Convert token spans back to characters using:

offset_mapping

Then extract:

text[char_start:char_end]

Never reconstruct using:

* tokenizer.decode()
* token joins
* whitespace normalization

Reason:
exact character fidelity matters.

⸻

18. MEMORY SAFETY RULES

The generated script MUST:

* delete fold-local tensors after use
* avoid storing unnecessary hidden states
* avoid storing attention maps
* avoid keeping duplicated datasets

Recommended:

* float32 logits only
* int32 token arrays

⸻

19. FAILURE DEFENSE RULES

The generated solution MUST explicitly guard against:

A. Empty span mappings

Fallback:

return full text

⸻

B. e_token < s_token

Fix:

e_token = s_token

⸻

C. NaN predictions

Validate:

np.isnan(...)

Abort if detected.

⸻

D. Dataset misalignment

Verify:

* feature lengths
* target lengths
* prediction lengths

at every stage.

⸻

20. SUBMISSION RULES

Final output MUST contain:

textID
selected_text

Save:

submission.csv

with:

index=False

⸻

21. IMPORTANT ANTI-PATTERNS

The generated solution MUST NEVER:

* use seq2seq generation
* use greedy text reconstruction
* decode with tokenizer.decode
* normalize whitespace globally
* strip tweets
* lowercase everything
* use first-match-only span alignment
* ensemble decoded substrings
* independently argmax start/end
* allow spans outside text segment
* train without stratification
* use plain KFold
* omit offset mappings
* omit sequence_ids filtering
* use unstable DDP setup
* train DeBERTa-large without checkpointing
* use FP16 hacks without stability checks
* use CRF/span postprocessing complexity
* use pseudo-labeling unless rigorously validated

⸻

22. RECOMMENDED OVERALL STRUCTURE

imports
global config
load_data()
custom splitter
span search utilities
preprocess()
dataset class
model class
checkpoint averaging
train worker
ddp orchestrator
ensemble()
workflow()
main

⸻

23. EXPECTED COMPETITIVE CHARACTERISTICS

A correctly generated implementation should exhibit:

* stable CV
* strong neutral handling
* high span fidelity
* low decoding variance
* robust fold agreement
* strong public/private LB correlation
* medal-level extractive QA behavior

The solution should prioritize:

stability > cleverness

because this competition rewards:

* exact substring reconstruction
* alignment correctness
* decoding reliability
* ensemble consistency

far more than exotic modeling tricks.