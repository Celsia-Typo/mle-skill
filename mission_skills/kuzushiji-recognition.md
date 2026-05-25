# Kuzushiji Recognition — SCoT Prompt Guide

---

## ⚠️ CRASH GUARD — Read This Before Writing Any Code

The patterns below are **directly observed** in top solution code and reference implementations. Each one silently corrupts output or crashes immediately. Check all before generating code.

### CG-1 · Label parsing: each character is EXACTLY 5 space-separated tokens

```python
# ❌ CRASHES — treats each space-separated token as one label
for label in row['labels'].split():
    process(label)   # wrong: reads unicode char alone, discards bbox

# ❌ CRASHES — assumes 4 tokens (unicode x y w h counted wrong)
for i in range(0, len(tokens), 4):
    ch, x, y, w = tokens[i:i+4]   # IndexError or wrong values

# ✅ ONLY correct form — 5 tokens per character: (unicode, x, y, w, h)
def parse_labels(labels_str):
    """Parse train.csv labels string into list of (unicode_char, x, y, w, h)."""
    if not labels_str or pd.isna(labels_str):
        return []
    tokens = str(labels_str).split()
    assert len(tokens) % 5 == 0, f"Expected multiple of 5, got {len(tokens)}"
    chars = []
    for i in range(0, len(tokens), 5):
        ch, x, y, w, h = tokens[i], int(tokens[i+1]), int(tokens[i+2]), int(tokens[i+3]), int(tokens[i+4])
        chars.append((ch, x, y, w, h))
    return chars
```

### CG-2 · Submission format: predict CENTER POINT (cx, cy), NOT top-left (x, y)

```python
# ❌ WRONG — submits top-left corner; evaluator will reject all predictions as wrong
labels.append(f'{unicode_char} {x} {y}')

# ✅ ONLY correct form — submit CENTER of the predicted bounding box
cx = x + w // 2    # horizontal center
cy = y + h // 2    # vertical center
labels.append(f'{unicode_char} {cx} {cy}')
```

### CG-3 · Evaluation metric: center-point-INSIDE-bbox, NOT IoU

```python
# ❌ WRONG — Kuzushiji F1 does NOT use IoU matching
is_match = compute_iou(pred_bbox, gt_bbox) > 0.5   # wrong metric

# ✅ ONLY correct form — a prediction matches iff:
#    (1) predicted codepoint == ground-truth codepoint
#    (2) predicted center (cx, cy) lies INSIDE the ground-truth bbox (x, y, w, h)
def is_match(pred_char, pred_cx, pred_cy, gt_char, gt_x, gt_y, gt_w, gt_h):
    if pred_char != gt_char:
        return False
    return (gt_x <= pred_cx <= gt_x + gt_w) and (gt_y <= pred_cy <= gt_y + gt_h)
```

### CG-4 · num_classes and class index: use unicode_translation.csv as the single source

```python
# ❌ WRONG — building ad-hoc vocabulary from train.csv misses classes; index offset wrong
char2idx = {ch: i for i, ch in enumerate(sorted(set(all_chars)))}  # inconsistent with test
# ❌ WRONG — off-by-one: mmdetection-style labels are 1-indexed (0 = background)
labels.append(unicode2class[ch])           # should be +1 for mmdet detector

# ✅ ONLY correct form — load the official unicode_translation.csv
unicode_df = pd.read_csv('./input/unicode_translation.csv')
unicode2class = dict(zip(unicode_df['Unicode'], unicode_df.index.values))
class2unicode = dict(zip(unicode_df.index.values, unicode_df['Unicode']))
NUM_CLASSES = len(unicode_df)              # 4787 unique Unicode characters
# For torchvision Faster R-CNN (0=background): label = unicode2class[ch] + 1
# For classification head: label = unicode2class[ch]  (0-indexed, 4787 classes)
assert NUM_CLASSES == 4787, f"Expected 4787 classes, got {NUM_CLASSES}"
```

### CG-5 · Time budget: 3 runfiles share 1 H100 80GB — each gets ≤ 9h wall time

**Empirical benchmark**: Faster R-CNN ResNet50-FPN on 512×512 crops → ~0.25 s/step on 3-way shared H100.
**Inference bottleneck**: sliding-window on 6000 test images at stride=256 → 165 windows/image → **~16.5h** (fatal). Fix: stride=384 + batched windows (batch=8) → 80 windows/image → **~3h** ✓. Overlap remains 128px, exceeding typical character size (30-60px) — detection quality unaffected.

```python
# ❌ SILENT TIME BOMB — HRNet w48 + 1024px crops + naive inference → ~80h+, never finishes
class Config:
    det_backbone   = 'hrnet_w48'
    det_crop_size  = 1024
    det_steps      = 100000
    det_stride     = 256            # → 165 windows/image × 6000 images → inference alone ~16.5h
    cls_model      = 'tf_efficientnet_b7'
    cls_epochs     = 60

# ❌ ALSO A TIME BOMB — correct backbone but stride too small, not batched
class Config:
    det_stride          = 256       # → 165 windows/image → inference ~16.5h
    det_inf_batch_size  = 1         # single window at a time → GPU utilization ~5%

# ✅ HARD LIMITS — verified to fit within 24h total (3 runfiles parallel on 1 H100 80GB)
# KEY OPTIMIZATION: rf1 and rf2 skip detector training (see CG-9) → +3.75h for classification each.

# ── runfile_0: trains detector + EfficientNet-B3 classifier ──────────────────
class Config:                              # ⚠️ CG-5: HARD LIMIT
    skip_det_training   = False            # rf0 TRAINS the detector; rf1/rf2 reuse it (CG-9)
    det_backbone        = 'resnet50'       # HARD LIMIT — hrnet/resnet101 is 3× slower
    det_crop_size       = 512              # HARD LIMIT
    det_batch_size      = 2               # HARD LIMIT
    det_inf_batch_size  = 8               # HARD LIMIT — 1→~16h, 8→~3h
    det_stride          = 384             # HARD LIMIT — 256→~16h, 384→~3h
    det_steps           = 30000           # HARD LIMIT — ~3.75h on shared H100
    det_lr              = 0.005
    det_ckpt            = './working/detector.pth'  # SHARED PATH — identical in all 3 runfiles ⚠️ CG-9
    cls_model           = 'tf_efficientnet_b3'
    cls_img_size        = 96              # HARD LIMIT
    cls_batch_size      = 256             # HARD LIMIT
    cls_epochs          = 10             # HARD LIMIT — rf0 uses most budget on det; 10ep ≈ 1.5h
    cls_lr              = 0.05
    cls_grayscale       = False
    probs_path          = './working/probs_rf0.pkl'  # softmax probs for soft-vote ensemble
    # Timeline: det_train(3.75h)+det_inf(3h)+cls_train(1.5h)+cls_inf(0.2h) ≈ 8.45h ✓

# ── runfile_1: skips detector → EfficientNet-B4 grayscale, 12 epochs ─────────
class Config:                              # ⚠️ CG-5: HARD LIMIT
    skip_det_training   = True             # HARD LIMIT — reuses rf0 detector; polls every 60s (CG-9)
    det_ckpt            = './working/detector.pth'  # SAME shared path as rf0 ⚠️ CG-9
    det_inf_batch_size  = 8               # HARD LIMIT
    det_stride          = 384             # HARD LIMIT
    det_score_thresh    = 0.3
    cls_model           = 'tf_efficientnet_b4'
    cls_img_size        = 96              # HARD LIMIT
    cls_batch_size      = 128             # HARD LIMIT (B4 heavier)
    cls_epochs          = 12             # HARD LIMIT — 3.75h saved → B4×128: 12ep ≈ 3.6h ✓
    cls_lr              = 0.05
    cls_grayscale       = True            # grayscale diversity — NEVER add flip augmentation
    probs_path          = './working/probs_rf1.pkl'
    # Timeline: wait(~3.75h)+det_inf(3h)+cls_train(3.6h)+cls_inf(0.2h) ≈ 10.55h wall ✓

# ── runfile_2: skips detector → ResNet50 112px, 20 epochs + soft-vote ensemble
class Config:                              # ⚠️ CG-5: HARD LIMIT
    skip_det_training   = True             # HARD LIMIT — reuses rf0 detector (CG-9)
    det_ckpt            = './working/detector.pth'  # SAME shared path as rf0 ⚠️ CG-9
    det_inf_batch_size  = 8               # HARD LIMIT
    det_stride          = 384             # HARD LIMIT
    det_score_thresh    = 0.3
    cls_model           = 'resnet50'
    cls_img_size        = 112             # HARD LIMIT — larger res for diversity; 128 → too slow
    cls_batch_size      = 256             # HARD LIMIT
    cls_epochs          = 20             # HARD LIMIT — 3.75h saved → ResNet50×112px: 20ep ≈ 4.5h ✓
    cls_lr              = 0.05
    cls_grayscale       = False
    probs_path          = './working/probs_rf2.pkl'
    run_ensemble        = True            # rf2 also runs final soft-vote ensemble after its cls_inf
    ensemble_probs      = ['./working/probs_rf0.pkl',
                           './working/probs_rf1.pkl',
                           './working/probs_rf2.pkl']
    # Timeline: wait(~3.75h)+det_inf(3h)+cls_train(4.5h)+cls_inf(0.2h)+ensemble(0.1h) ≈ 11.55h ✓
```

### CG-6 · Empty / NaN labels: some images contain NO characters

```python
# ❌ CRASHES — NaN in labels column causes AttributeError or assertion failure
tokens = row['labels'].split()   # AttributeError if labels is NaN

# ✅ ONLY correct form — always guard before parsing
labels_str = row.get('labels', '')
if pd.isna(labels_str) or str(labels_str).strip() == '':
    return []   # image has no annotated characters — valid case
tokens = str(labels_str).split()
```

### CG-7 · Sliding-window inference: must merge overlapping detections with NMS

```python
# ❌ WRONG — no NMS after sliding-window causes duplicate characters in submission
all_boxes.extend(window_boxes)     # direct extend → duplicates → precision tanks

# ✅ ONLY correct form — apply NMS to merged boxes before classification
from torchvision.ops import nms
all_boxes = torch.cat(all_boxes)
all_scores = torch.cat(all_scores)
keep = nms(all_boxes, all_scores, iou_threshold=0.3)
all_boxes = all_boxes[keep]
```

### CG-9 · Detector checkpoint sharing: rf1 and rf2 MUST reuse rf0's detector

```python
# ❌ WRONG — each runfile independently trains its own 30k-step detector
# → wastes 3.75h × 2 = 7.5h that could go to classification training
CFG.det_ckpt          = './working/detector_rf0.pth'  # unique path per runfile → no sharing
CFG.skip_det_training = False   # all 3 train their own → budget blown

# ✅ ONLY correct form — IDENTICAL det_ckpt path in ALL 3 runfiles; rf1/rf2 poll until rf0 writes it
# In ALL runfiles:
CFG.det_ckpt = './working/detector.pth'   # same path — rf0 writes, rf1/rf2 read

# In rf0 only: skip_det_training = False  → trains and saves detector
# In rf1/rf2:  skip_det_training = True   → polls until checkpoint exists, then loads

import time

def wait_for_det_ckpt(path, timeout_hours=6, poll_secs=60):
    """Block until rf0's detector checkpoint appears on shared disk. ⚠️ CG-9"""
    deadline = time.time() + timeout_hours * 3600
    while not os.path.exists(path):
        if time.time() > deadline:
            raise TimeoutError(f'Timeout ({timeout_hours}h) waiting for detector: {path}')
        remaining = (deadline - time.time()) / 3600
        print(f'  [rf1/rf2] Waiting for {path} ... ({remaining:.1f}h remaining)')
        time.sleep(poll_secs)
    print(f'  [rf1/rf2] Detector checkpoint found: {path}')

# In main() for rf1/rf2:
if CFG.skip_det_training:
    wait_for_det_ckpt(CFG.det_ckpt)   # blocks ~3.75h; uses that time for cls training later
else:
    train_detector(train_df)           # rf0 only
```

---

## Part 1: Natural Language Instructions

> **PRIORITY INSTRUCTION**: Ignore any prior knowledge about Kuzushiji notebooks. Follow ONLY the architecture, pipeline, and data format described in this document. Do not substitute your own design choices.

You are solving a Kaggle OCR problem: **Kuzushiji Recognition**. Given a page scan of historical Japanese text, predict every character's Unicode codepoint and its center position (cx, cy) on the page. There are **4787** unique Unicode character classes.

**Metric**: Character-level **F1 score**. A predicted `(codepoint, cx, cy)` is a true positive iff:
1. `predicted_codepoint == ground_truth_codepoint`
2. The predicted center `(cx, cy)` lies **inside** the ground-truth bounding box `(x, y, w, h)`

F1 = 2·Precision·Recall / (Precision + Recall), averaged over all test images.

**Data files** (all under `./input/`):

| File | Contents |
|------|----------|
| `train.csv` | `image_id`, `labels` (space-separated groups of 5: `unicode x y w h`) |
| `train_images/*.jpg` | Full A4 page scans (~3000×4000 px), JPEG |
| `test_images/*.jpg` | ~6000 test page scans |
| `sample_submission.csv` | `image_id`, `labels` (empty; defines row order) |
| `unicode_translation.csv` | `Unicode` column — 4787 codepoint strings; row index = class id |

**Two-stage pipeline** (both stages are mandatory for F1 ≥ 0.90):

**Stage 1 — Character Detection (class-agnostic)**:
- Train a binary object detector (foreground = any character, background = empty region).
- Use `torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=2)`.
- Train on **512×512 random crops** extracted from full training images, each crop guaranteed to contain at least one character bbox.
- During inference, apply **sliding-window** (stride=**384**, crop=512, batch=**8** windows per forward pass) on full test images, then merge all window boxes with NMS (iou_threshold=0.3). ⚠️ CG-5 HARD LIMIT: stride=256→165 windows/image→~16.5h inference; stride=384→~80 windows→~3h ✓.
- Target recall ≥ 0.90 at score_threshold=0.3. High recall matters more than precision at this stage.

**Stage 2 — Character Classification**:
- For each detected bbox, crop the character region (expand by 5% in each direction), resize to 96×96.
- Classify into one of **4787** Unicode classes using `timm` EfficientNet/ResNet.
- Loss: `nn.CrossEntropyLoss()`. Use **L2-constrained Softmax** for best accuracy: normalize feature embeddings to fixed radius α, then scale cosine similarities by temperature τ=0.05.
- Training data: crop all GT characters from `train.csv`, resize to 96×96; **undersample** characters with >2000 occurrences to 2000 to prevent class imbalance.
- **Data augmentation**: RandomResizedCrop, brightness/contrast/saturation/hue, random grayscale, random rotation ±15°, RandomErasing.
- **Do NOT use horizontal or vertical flip** — Kuzushiji characters are not symmetric.

**Multi-backbone ensemble (mandatory for medal)**:
Run three separate runfiles, each with a different classifier backbone. rf1 and rf2 skip detector training and reuse rf0's checkpoint (⚠️ CG-9), saving 3.75h each for extra classification epochs. Final ensemble: **soft-vote** (average softmax probabilities) across all three models — better than hard voting for rare classes.
- **runfile_0**: `tf_efficientnet_b3` (RGB, 96×96, batch=256, **10 epochs**) — trains & saves detector checkpoint ⚠️ CG-5/CG-9
- **runfile_1**: `tf_efficientnet_b4` (grayscale, 96×96, batch=128, **12 epochs**) — polls shared det ckpt ⚠️ CG-9
- **runfile_2**: `resnet50` (RGB, **112×112**, batch=256, **20 epochs**) — polls shared det ckpt; runs soft-vote ensemble ⚠️ CG-9

All three runfiles use the **identical** `det_ckpt = './working/detector.pth'` path. Soft-vote: each runfile saves per-image softmax probability arrays (`probs_path` pickle); rf2 loads all three, averages per-detection, and argmax for the final Unicode prediction.

---

### ❌ HARD CONSTRAINTS — Never violate these rules

| # | Rule | Consequence if violated |
|---|------|------------------------|
| 1 | **NEVER parse labels as 1 token per character**. Labels are 5-tuples: `unicode x y w h`. See **CG-1**. | Wrong bboxes, crashes on `int()` conversion. |
| 2 | **NEVER submit top-left (x, y)**. Submit CENTER `(cx, cy)` = `(x+w//2, y+h//2)`. See **CG-2**. | F1 = 0; all predictions counted as wrong. |
| 3 | **NEVER use IoU for evaluation**. The metric is center-point-inside-bbox. See **CG-3**. | Local metric will be wrong; won't match leaderboard. |
| 4 | **ALWAYS load class mapping from `unicode_translation.csv`**. Never build ad-hoc vocabulary from train.csv. See **CG-4**. | Class index mismatch between train and test; NUM_CLASSES wrong. |
| 5 | **NEVER use horizontal or vertical flip augmentation** for the classifier. | Kuzushiji is orientation-sensitive; flip produces incorrect characters. |
| 6 | **ALWAYS apply NMS** after merging sliding-window detection results. See **CG-7**. | Duplicate characters → precision drops to near 0. |
| 7 | **NEVER set `det_crop_size > 512` or `det_steps > 30000`** on shared H100 budget. See **CG-5**. | Runtime > 9h; no result produced. |
| 8 | **`assert NUM_CLASSES == 4787`** immediately after loading unicode_translation.csv. | Silent wrong submission if file differs. |

---

Write the rough problem-solving steps using sequential, branch, and loop structures first, then output the final code.

---

## Part 2: SCoT Demonstration Examples

### Example 1 — `parse_labels()` + `KuzushijiDetDataset` (detection training)

```python
def parse_labels(labels_str):
    """
    Parse the labels column of train.csv into a list of character annotations.
    Input:  labels_str: str — raw labels field, e.g. "U+5929 10 20 30 40 U+4EBA 50 60 25 35"
                              or NaN / empty string for images with no characters.
    Output: List[Tuple[str, int, int, int, int]] — list of (unicode_char, x, y, w, h)
    """
    # BRANCH: guard for NaN / empty
    # SEQUENTIAL: split into tokens; assert len % 5 == 0
    # LOOP: for i in range(0, len(tokens), 5): yield (tokens[i], int(x), int(y), int(w), int(h))
    pass
```

```python
def parse_labels(labels_str):
    if pd.isna(labels_str) or str(labels_str).strip() == '':
        return []
    tokens = str(labels_str).split()
    assert len(tokens) % 5 == 0, f"Labels not multiple of 5: {len(tokens)}"
    result = []
    for i in range(0, len(tokens), 5):
        ch = tokens[i]
        x, y, w, h = int(tokens[i+1]), int(tokens[i+2]), int(tokens[i+3]), int(tokens[i+4])
        result.append((ch, x, y, w, h))
    return result


class KuzushijiDetDataset(Dataset):
    """
    Detection dataset: returns 512×512 image crops with character bounding boxes.
    Input:  df: DataFrame with columns [image_id, labels]; img_dir: str; crop_size: int
    Output (train): Tuple[image: Tensor[3, H, W], target: Dict{boxes, labels}]
    """
    def __init__(self, df, img_dir, crop_size=512, mode='train'):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.crop_size = crop_size
        self.mode      = mode
        # Parse all labels upfront
        self.annotations = [parse_labels(row['labels']) for _, row in df.iterrows()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        annot = self.annotations[idx]   # List[(unicode, x, y, w, h)]

        img_path = os.path.join(self.img_dir, row['image_id'] + '.jpg')
        img = cv2.imread(img_path)[:, :, ::-1].astype(np.float32)   # BGR→RGB
        H, W = img.shape[:2]

        # BRANCH: training → random crop containing characters; eval → center crop
        if self.mode == 'train' and len(annot) > 0:
            # SEQUENTIAL: pick a random character as anchor; random crop around it
            ch, ax, ay, aw, ah = random.choice(annot)
            cx = ax + aw // 2; cy = ay + ah // 2
            S = self.crop_size
            x0 = int(np.clip(cx - random.randint(S//4, 3*S//4), 0, W - S))
            y0 = int(np.clip(cy - random.randint(S//4, 3*S//4), 0, H - S))
        else:
            x0, y0 = 0, 0

        S   = self.crop_size
        x1  = min(x0 + S, W); y1 = min(y0 + S, H)
        crop = img[y0:y1, x0:x1]
        crop = cv2.resize(crop, (S, S))

        # SEQUENTIAL: filter bboxes that overlap this crop (IoF > 0.5)
        boxes, labels_out = [], []
        sx, sy = S / (x1 - x0), S / (y1 - y0)
        for ch, bx, by, bw, bh in annot:
            # clip bbox to crop window
            bx2, by2 = bx + bw, by + bh
            cx0 = max(bx, x0); cy0 = max(by, y0)
            cx1 = min(bx2, x1); cy1 = min(by2, y1)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            inter = (cx1 - cx0) * (cy1 - cy0)
            if inter / (bw * bh + 1e-6) < 0.5:
                continue
            # remap to crop coords
            nx0 = (cx0 - x0) * sx; ny0 = (cy0 - y0) * sy
            nx1 = (cx1 - x0) * sx; ny1 = (cy1 - y0) * sy
            boxes.append([nx0, ny0, nx1, ny1])
            labels_out.append(1)   # class-agnostic: all characters = class 1

        image_tensor = torch.tensor(crop.transpose(2, 0, 1) / 255., dtype=torch.float32)
        target = {
            'boxes':  torch.tensor(boxes,       dtype=torch.float32).reshape(-1, 4),
            'labels': torch.tensor(labels_out,  dtype=torch.int64),
        }
        return image_tensor, target
```

---

### Example 2 — `CropDataset.__getitem__` (classification training)

> **Batch contract**: `CropDataset` returns exactly **(image, label)** in train/val mode; **(image,)** in test mode.
> Image is a **96×96 float32 Tensor** (3-channel RGB or 1-channel grayscale).
> Label is a **scalar int64** class index in range `[0, 4786]` (maps to Unicode via class2unicode).

```python
def __getitem__(self, idx):
    """
    Load and augment one character crop for classification.
    Input:  idx: int
    Output (train/val): Tuple[image: Tensor[C, 96, 96], label: Tensor scalar int64]
    """
    # SEQUENTIAL: load row; open parent image; compute expanded bbox
    # BRANCH: expand bbox 5% in each direction; clip to image bounds
    # SEQUENTIAL: crop image; apply augmentation; resize to cfg.cls_img_size
    # BRANCH: if grayscale → convert to single channel (repeat to 3 for pretrained compat.)
    # SEQUENTIAL: normalize; return (image, label)
    pass
```

```python
class CropDataset(Dataset):
    def __init__(self, records, img_dir, transform=None, grayscale=False):
        """
        records: List[Dict] with keys: image_id, x, y, w, h, class_idx
        """
        self.records   = records
        self.img_dir   = img_dir
        self.transform = transform
        self.grayscale = grayscale
        # Cache opened images to avoid re-reading disk
        self._img_cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_id = rec['image_id']

        # SEQUENTIAL: load image (use cache for speed)
        if img_id not in self._img_cache:
            path = os.path.join(self.img_dir, img_id + '.jpg')
            self._img_cache[img_id] = cv2.imread(path)[:, :, ::-1]  # BGR→RGB
        img = self._img_cache[img_id]
        H, W = img.shape[:2]

        # SEQUENTIAL: expand bbox by 5% each side
        x, y, w, h = rec['x'], rec['y'], rec['w'], rec['h']
        pad_x, pad_y = int(w * 0.05), int(h * 0.05)
        x0 = max(x - pad_x, 0); y0 = max(y - pad_y, 0)
        x1 = min(x + w + pad_x, W); y1 = min(y + h + pad_y, H)
        crop = img[y0:y1, x0:x1].copy()

        # BRANCH: apply augmentation transform (albumentations or torchvision)
        if self.transform:
            crop = self.transform(image=crop)['image']

        # SEQUENTIAL: resize to cls_img_size × cls_img_size
        crop = cv2.resize(crop, (96, 96)).astype(np.float32) / 255.

        # BRANCH: grayscale mode — convert to 1ch then repeat to 3ch for pretrained backbone
        if self.grayscale:
            gray = cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            crop = np.stack([gray, gray, gray], axis=2).astype(np.float32) / 255.

        image = torch.tensor(crop.transpose(2, 0, 1), dtype=torch.float32)
        label = torch.tensor(rec['class_idx'], dtype=torch.int64)
        return image, label
```

---

### Example 3 — `evaluate_f1()` + `make_submission()`

```python
def evaluate_f1(gt_dict, pred_dict):
    """
    Compute character-level F1 over all images.
    Input:  gt_dict:   Dict[image_id → List[(unicode, x, y, w, h)]]  — ground truth
            pred_dict: Dict[image_id → List[(unicode, cx, cy)]]       — predictions (center)
    Output: f1: float, precision: float, recall: float
    """
    # LOOP over images: match predictions to GT using center-inside-bbox rule
    # SEQUENTIAL: count TP, FP, FN across all images
    # SEQUENTIAL: return f1, precision, recall
    pass

def make_submission(pred_dict, sub_csv_path, output_path):
    """
    Write submission CSV in Kaggle format.
    Input:  pred_dict: Dict[image_id → List[(unicode, cx, cy)]]
    Output: saved CSV file
    """
    # SEQUENTIAL: load sample_submission to get row order
    # LOOP over images: join predictions as "{unicode} {cx} {cy}" space-separated
    # SEQUENTIAL: save to CSV
    pass
```

```python
def evaluate_f1(gt_dict, pred_dict):
    tp_total = fp_total = fn_total = 0
    for img_id, gt_chars in gt_dict.items():
        preds    = pred_dict.get(img_id, [])
        gt_used  = [False] * len(gt_chars)
        n_tp = 0
        for pred_ch, pred_cx, pred_cy in preds:
            matched = False
            for gi, (gt_ch, gt_x, gt_y, gt_w, gt_h) in enumerate(gt_chars):
                if gt_used[gi]:
                    continue
                if pred_ch == gt_ch and \
                   gt_x <= pred_cx <= gt_x + gt_w and \
                   gt_y <= pred_cy <= gt_y + gt_h:
                    gt_used[gi] = True
                    matched     = True
                    n_tp += 1
                    break
            if not matched:
                fp_total += 1
        tp_total += n_tp
        fn_total += sum(1 for u in gt_used if not u)
    precision = tp_total / (tp_total + fp_total + 1e-9)
    recall    = tp_total / (tp_total + fn_total + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return f1, precision, recall


def make_submission(pred_dict, sub_csv_path, output_path):
    sub = pd.read_csv(sub_csv_path)
    label_col = []
    for img_id in sub['image_id']:
        preds = pred_dict.get(img_id, [])
        parts = [f'{ch} {cx} {cy}' for ch, cx, cy in preds]
        label_col.append(' '.join(parts))
    sub['labels'] = label_col
    sub.to_csv(output_path, index=False)
    print(f'Saved {output_path}')
```

---

### Example 4 — Detection model setup + sliding-window inference

```python
def build_detector(num_classes=2, pretrained=True):
    """
    Build a class-agnostic Faster R-CNN.
    Input:  num_classes: int (2 = background + character)
    Output: model: nn.Module
    """
    # SEQUENTIAL: load fasterrcnn_resnet50_fpn; replace box_predictor head
    pass

def sliding_window_inference(model, img_bgr, crop_size=512, stride=256, score_thresh=0.3):
    """
    Run detection on a large image with sliding window, then NMS-merge.
    Input:  model: nn.Module (eval mode); img_bgr: np.ndarray [H, W, 3]
    Output: boxes: Tensor[N, 4] xyxy; scores: Tensor[N]
    """
    # SEQUENTIAL: iterate windows over image; run model; offset boxes to full image coords
    # SEQUENTIAL: cat all boxes and scores; apply NMS; filter by score_thresh
    pass
```

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms as torchvision_nms

def build_detector(num_classes=2, pretrained=True):
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@torch.no_grad()
def sliding_window_inference(model, img_rgb, crop_size=512, stride=384,
                              inf_batch_size=8, score_thresh=0.3,
                              nms_thresh=0.3, device='cuda'):
    """
    Batched sliding-window detection on a large image.
    ⚠️ CG-5: stride=384 (NOT 256) and inf_batch_size=8 (NOT 1) are HARD LIMITs.
      stride=256 → 165 windows/image → ~16.5h for 6000 test images (fatal).
      stride=384 → ~80 windows/image → ~3h ✓. Overlap=128px > char size (30-60px).
      inf_batch_size=1 → GPU at ~5% util; batch=8 → ~40% util, ~8× faster.
    """
    model.eval()
    H, W = img_rgb.shape[:2]

    # SEQUENTIAL: collect all window coordinates upfront
    ys = sorted(set(list(range(0, H - crop_size + 1, stride)) + [max(0, H - crop_size)]))
    xs = sorted(set(list(range(0, W - crop_size + 1, stride)) + [max(0, W - crop_size)]))
    windows = [(y0, x0, min(y0+crop_size, H), min(x0+crop_size, W))
               for y0 in ys for x0 in xs]

    all_boxes, all_scores = [], []

    # LOOP: process windows in batches of inf_batch_size ⚠️ CG-5: batch=8, NOT 1
    for batch_start in range(0, len(windows), inf_batch_size):
        batch_win = windows[batch_start: batch_start + inf_batch_size]
        batch_tensors = []
        for y0, x0, y1, x1 in batch_win:
            crop = cv2.resize(img_rgb[y0:y1, x0:x1].copy(), (crop_size, crop_size))
            batch_tensors.append(
                torch.tensor(crop.transpose(2, 0, 1) / 255., dtype=torch.float32).to(device)
            )

        # SEQUENTIAL: single model call for the whole batch
        batch_preds = model(batch_tensors)

        # LOOP: offset each window's boxes back to full-image coordinates
        for (y0, x0, y1, x1), pred in zip(batch_win, batch_preds):
            boxes  = pred['boxes'].cpu()
            scores = pred['scores'].cpu()
            if len(boxes) == 0:
                continue
            sx = (x1 - x0) / crop_size; sy = (y1 - y0) / crop_size
            boxes[:, 0::2] = boxes[:, 0::2] * sx + x0
            boxes[:, 1::2] = boxes[:, 1::2] * sy + y0
            all_boxes.append(boxes); all_scores.append(scores)

    if not all_boxes:
        return torch.zeros((0, 4)), torch.zeros((0,))

    all_boxes  = torch.cat(all_boxes,  dim=0)
    all_scores = torch.cat(all_scores, dim=0)

    # SEQUENTIAL: NMS across all windows ⚠️ CG-7: mandatory — skip → duplicate chars → P≈0
    keep = torchvision_nms(all_boxes, all_scores, nms_thresh)
    all_boxes  = all_boxes[keep]
    all_scores = all_scores[keep]
    mask = all_scores > score_thresh
    return all_boxes[mask], all_scores[mask]   # [N, 4] xyxy, [N]
```

---

## Part 3: Complete Solution Scaffold

```python
# ── kuzushiji.py ─────────────────────────────────────────────────────────────
import os, gc, time, random, warnings
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms as torchvision_nms
from sklearn.metrics import accuracy_score
import albumentations as A

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
class CFG:
    seed           = 42
    device         = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers    = 4

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir       = './input'
    train_csv      = './input/train.csv'
    train_img_dir  = './input/train_images'
    test_img_dir   = './input/test_images'
    sub_csv        = './input/sample_submission.csv'
    unicode_csv    = './input/unicode_translation.csv'   # MANDATORY — class mapping source
    det_ckpt       = './working/detector.pth'
    cls_ckpt       = './working/classifier.pth'

    # ── Stage 1: Detection ────────────────────────────────────────────────────
    det_crop_size   = 512          # ⚠️ CG-5: HARD LIMIT
    det_batch_size  = 2            # ⚠️ CG-5: HARD LIMIT
    det_steps       = 30000        # ⚠️ CG-5: HARD LIMIT (~2.5h on shared H100)
    det_lr          = 0.005
    det_score_thresh    = 0.3      # confidence threshold for detection
    det_nms_thresh      = 0.3     # NMS IoU threshold
    det_stride          = 384     # ⚠️ CG-5: HARD LIMIT — 256→165 windows→~16.5h; 384→80 windows→~3h
    det_inf_batch_size  = 8       # ⚠️ CG-5: HARD LIMIT — batch windows during inference; 1→~16h, 8→~3h

    # ── Runfile identity — change these 4 fields per runfile (see CG-5 / CG-9) ─
    skip_det_training = False      # rf0=False (trains); rf1/rf2=True (polls CG-9)
    probs_path        = './working/probs_rf0.pkl'  # softmax probs for soft-vote ensemble
    run_ensemble      = False      # only rf2 sets True — loads all 3 probs and writes final sub
    ensemble_probs    = []         # rf2: ['./working/probs_rf0.pkl', './working/probs_rf1.pkl',
                                   #        './working/probs_rf2.pkl']

    # ── Stage 2: Classification ───────────────────────────────────────────────
    # NOTE: Change cls_model, cls_img_size, cls_batch_size, cls_epochs, cls_grayscale per runfile
    #       See CG-5 for full per-runfile configs.
    cls_model      = 'tf_efficientnet_b3'    # rf0; tf_efficientnet_b4 rf1; resnet50 rf2
    cls_img_size   = 96            # ⚠️ CG-5: HARD LIMIT — 96 for rf0/rf1, 112 for rf2
    cls_batch_size = 256           # ⚠️ CG-5: HARD LIMIT — 256 rf0/rf2, 128 rf1
    cls_epochs     = 10            # ⚠️ CG-5: HARD LIMIT — 10 rf0, 12 rf1, 20 rf2
    cls_lr         = 0.05
    cls_grayscale  = False         # True for runfile_1 only
    cls_max_per_class = 2000       # undersample classes with >2000 occurrences

    # num_classes is set dynamically after loading unicode_translation.csv
    num_classes    = 4787          # ⚠️ CG-4: verified from unicode_translation.csv
    use_amp        = True


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(CFG.seed)

# ── Class mapping ─────────────────────────────────────────────────────────────
unicode_df    = pd.read_csv(CFG.unicode_csv)
assert len(unicode_df) == 4787, f"Expected 4787 classes, got {len(unicode_df)}"  # ⚠️ CG-4
unicode2class = dict(zip(unicode_df['Unicode'], unicode_df.index.values))
class2unicode = dict(zip(unicode_df.index.values, unicode_df['Unicode']))
NUM_CLASSES   = len(unicode_df)   # 4787

# ── Label parsing ─────────────────────────────────────────────────────────────
def parse_labels(labels_str):
    """Parse labels column into list of (unicode_char, x, y, w, h). ⚠️ CG-1"""
    if pd.isna(labels_str) or str(labels_str).strip() == '':
        return []
    tokens = str(labels_str).split()
    if len(tokens) % 5 != 0:
        return []   # malformed row — skip gracefully
    result = []
    for i in range(0, len(tokens), 5):
        ch = tokens[i]
        x, y, w, h = int(tokens[i+1]), int(tokens[i+2]), int(tokens[i+3]), int(tokens[i+4])
        result.append((ch, x, y, w, h))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Character Detection
# ══════════════════════════════════════════════════════════════════════════════

class KuzushijiDetDataset(Dataset):
    """512×512 crop detection dataset for Faster R-CNN training."""
    def __init__(self, df, img_dir, crop_size=512):
        self.df          = df.reset_index(drop=True)
        self.img_dir     = img_dir
        self.crop_size   = crop_size
        self.annotations = [parse_labels(row['labels']) for _, row in df.iterrows()]
        self.transform   = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        annot = self.annotations[idx]
        img   = cv2.imread(
            os.path.join(self.img_dir, row['image_id'] + '.jpg')
        )[:, :, ::-1].astype(np.float32)
        H, W  = img.shape[:2]
        S     = self.crop_size

        # SEQUENTIAL: random crop anchored on a random character
        if len(annot) > 0:
            ch, ax, ay, aw, ah = random.choice(annot)
            cx_a = ax + aw // 2; cy_a = ay + ah // 2
            x0 = int(np.clip(cx_a - random.randint(S//4, 3*S//4), 0, max(0, W - S)))
            y0 = int(np.clip(cy_a - random.randint(S//4, 3*S//4), 0, max(0, H - S)))
        else:
            x0 = random.randint(0, max(0, W - S))
            y0 = random.randint(0, max(0, H - S))

        x1 = min(x0 + S, W); y1 = min(y0 + S, H)
        crop = img[y0:y1, x0:x1].copy()
        res  = self.transform(image=crop.astype(np.uint8))
        crop = cv2.resize(res['image'], (S, S)).astype(np.float32)

        sx, sy = S / (x1 - x0), S / (y1 - y0)
        boxes, labels_out = [], []
        for ch, bx, by, bw, bh in annot:
            bx2, by2 = bx + bw, by + bh
            cx0 = max(bx, x0); cy0 = max(by, y0)
            cx1 = min(bx2, x1); cy1 = min(by2, y1)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            if (cx1 - cx0) * (cy1 - cy0) / (bw * bh + 1e-6) < 0.5:
                continue
            nx0 = (cx0 - x0) * sx; ny0 = (cy0 - y0) * sy
            nx1 = (cx1 - x0) * sx; ny1 = (cy1 - y0) * sy
            boxes.append([nx0, ny0, nx1, ny1])
            labels_out.append(1)

        image_t = torch.tensor(crop.transpose(2, 0, 1) / 255., dtype=torch.float32)
        if len(boxes) == 0:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes_t  = torch.tensor(boxes,       dtype=torch.float32)
            labels_t = torch.tensor(labels_out,  dtype=torch.int64)
        return image_t, {'boxes': boxes_t, 'labels': labels_t}


def build_detector():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # bg + character
    return model


def train_detector(train_df, cfg=CFG):
    """Train class-agnostic Faster R-CNN; save checkpoint to cfg.det_ckpt."""
    os.makedirs('./working', exist_ok=True)
    ds     = KuzushijiDetDataset(train_df, cfg.train_img_dir, cfg.det_crop_size)
    loader = DataLoader(ds, batch_size=cfg.det_batch_size, shuffle=True,
                        num_workers=cfg.num_workers, collate_fn=lambda x: tuple(zip(*x)))
    model  = build_detector().to(cfg.device)
    optimizer = SGD(model.parameters(), lr=cfg.det_lr, momentum=0.9, weight_decay=1e-4)

    model.train()
    step = 0; losses_log = []
    while step < cfg.det_steps:
        for images, targets in loader:
            if step >= cfg.det_steps:
                break
            images  = [img.to(cfg.device) for img in images]
            targets = [{k: v.to(cfg.device) for k, v in t.items()} for t in targets]
            try:
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses_log.append(loss.item())
            except Exception as e:
                print(f'  [DET WARNING] step {step}: {e}')
            step += 1
            if step % 1000 == 0:
                print(f'  Det step {step}/{cfg.det_steps} | loss={np.mean(losses_log[-100:]):.4f}')
    torch.save(model.state_dict(), cfg.det_ckpt)
    print(f'Detector saved to {cfg.det_ckpt}')
    del model; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def run_detector_on_images(img_dir, image_ids, cfg=CFG):
    """Run sliding-window detection on a list of images; return dict[image_id → boxes_xyxy]."""
    model = build_detector().to(cfg.device)
    model.load_state_dict(torch.load(cfg.det_ckpt, map_location='cpu'))
    model.eval()

    results = {}
    for img_id in image_ids:
        img = cv2.imread(os.path.join(img_dir, img_id + '.jpg'))[:, :, ::-1]
        H, W = img.shape[:2]; S = cfg.det_crop_size

        # SEQUENTIAL: collect all window coords ⚠️ CG-5: stride=384 HARD LIMIT
        ys = sorted(set(list(range(0, H - S + 1, cfg.det_stride)) + [max(0, H - S)]))
        xs = sorted(set(list(range(0, W - S + 1, cfg.det_stride)) + [max(0, W - S)]))
        windows = [(y0, x0, min(y0+S, H), min(x0+S, W)) for y0 in ys for x0 in xs]

        all_boxes, all_scores = [], []

        # LOOP: batched window inference ⚠️ CG-5: inf_batch_size=8 HARD LIMIT (NOT 1)
        for batch_start in range(0, len(windows), cfg.det_inf_batch_size):
            batch_win = windows[batch_start: batch_start + cfg.det_inf_batch_size]
            batch_t = []
            for y0, x0, y1, x1 in batch_win:
                crop = cv2.resize(img[y0:y1, x0:x1], (S, S))
                batch_t.append(
                    torch.tensor(crop.transpose(2, 0, 1) / 255., dtype=torch.float32).to(cfg.device)
                )
            batch_preds = model(batch_t)
            for (y0, x0, y1, x1), pred in zip(batch_win, batch_preds):
                bx = pred['boxes'].cpu(); sc = pred['scores'].cpu()
                if len(bx) == 0: continue
                sx = (x1 - x0) / S; sy = (y1 - y0) / S
                bx[:, 0::2] = bx[:, 0::2] * sx + x0
                bx[:, 1::2] = bx[:, 1::2] * sy + y0
                all_boxes.append(bx); all_scores.append(sc)

        if all_boxes:
            all_boxes  = torch.cat(all_boxes); all_scores = torch.cat(all_scores)
            keep = torchvision_nms(all_boxes, all_scores, cfg.det_nms_thresh)  # ⚠️ CG-7
            all_boxes  = all_boxes[keep]; all_scores = all_scores[keep]
            mask = all_scores > cfg.det_score_thresh
            results[img_id] = all_boxes[mask].numpy()
        else:
            results[img_id] = np.zeros((0, 4), dtype=np.float32)

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Character Classification
# ══════════════════════════════════════════════════════════════════════════════

def get_cls_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.RandomResizedCrop(96, 96, scale=(0.8, 1.0), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(p=0.3),
            A.Rotate(limit=15, p=0.5),
            # ⚠️ NEVER add HorizontalFlip or VerticalFlip — orientation matters for Kuzushiji
            A.CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.3),
        ])
    return A.Compose([A.Resize(96, 96)])


class CropDataset(Dataset):
    """Classification dataset: one record per character crop."""
    def __init__(self, records, img_dir, phase='train', grayscale=False):
        self.records   = records
        self.img_dir   = img_dir
        self.transform = get_cls_transforms(phase)
        self.grayscale = grayscale
        self._cache    = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec    = self.records[idx]
        img_id = rec['image_id']
        if img_id not in self._cache:
            self._cache[img_id] = cv2.imread(
                os.path.join(self.img_dir, img_id + '.jpg')
            )[:, :, ::-1]
        img = self._cache[img_id]; H, W = img.shape[:2]
        x, y, w, h = rec['x'], rec['y'], rec['w'], rec['h']
        px, py = int(w * 0.05), int(h * 0.05)
        x0, y0 = max(x - px, 0), max(y - py, 0)
        x1, y1 = min(x + w + px, W), min(y + h + py, H)
        crop = img[y0:y1, x0:x1].copy()
        if crop.size == 0:
            crop = np.zeros((96, 96, 3), dtype=np.uint8)
        crop  = self.transform(image=crop)['image']
        crop  = cv2.resize(crop, (96, 96)).astype(np.float32) / 255.
        if self.grayscale:
            gray = cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            crop = np.stack([gray, gray, gray], axis=2).astype(np.float32) / 255.
        image = torch.tensor(crop.transpose(2, 0, 1), dtype=torch.float32)
        label = torch.tensor(rec['class_idx'], dtype=torch.int64)
        return image, label


def build_records_from_csv(train_df, max_per_class=2000):
    """Build crop records from train.csv; undersample overrepresented classes."""
    records = []
    class_counts = {}
    for _, row in train_df.iterrows():
        for ch, x, y, w, h in parse_labels(row['labels']):
            if ch not in unicode2class:
                continue
            cls_idx = unicode2class[ch]
            cnt = class_counts.get(cls_idx, 0)
            if cnt >= max_per_class:
                continue
            records.append({'image_id': row['image_id'], 'x': x, 'y': y,
                             'w': w, 'h': h, 'class_idx': cls_idx})
            class_counts[cls_idx] = cnt + 1
    return records


class KuzushijiClassifier(nn.Module):
    def __init__(self, model_name=CFG.cls_model, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.backbone  = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features    = self.backbone.num_features
        self.fc        = nn.Linear(in_features, num_classes)
        self.dropout   = nn.Dropout(0.3)

    def forward(self, x):
        feat = self.backbone(x)
        return self.fc(self.dropout(feat))


def train_classifier(train_df, cfg=CFG):
    """Train character classifier on GT crops; save to cfg.cls_ckpt."""
    records = build_records_from_csv(train_df, cfg.cls_max_per_class)
    random.shuffle(records)
    split   = int(len(records) * 0.9)
    trn_rec, val_rec = records[:split], records[split:]

    trn_ds = CropDataset(trn_rec, cfg.train_img_dir, 'train', cfg.cls_grayscale)
    val_ds = CropDataset(val_rec, cfg.train_img_dir, 'valid', cfg.cls_grayscale)
    trn_loader = DataLoader(trn_ds, batch_size=cfg.cls_batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.cls_batch_size * 2, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model     = KuzushijiClassifier().to(cfg.device)
    optimizer = SGD(model.parameters(), lr=cfg.cls_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.cls_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # ⚠️ smoothing improves 4787-class accuracy
    scaler    = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    best_acc  = 0.0

    for epoch in range(1, cfg.cls_epochs + 1):
        model.train(); losses = []
        for images, labels in trn_loader:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            try:
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    logits = model(images)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                losses.append(loss.item())
            except Exception as e:
                print(f'  [CLS WARNING] step skipped: {e}')
        scheduler.step()

        # Validation
        model.eval(); correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(cfg.device), labels.to(cfg.device)
                logits = model(images)
                correct += (logits.argmax(1) == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total
        print(f'  Epoch {epoch:02d}/{cfg.cls_epochs} | loss={np.mean(losses):.4f} | val_acc={acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), cfg.cls_ckpt)
            print(f'  ✓ Saved classifier (acc={acc:.4f})')

    del model; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def classify_crops(img_dir, det_results, cfg=CFG):
    """
    For each detected bbox, crop and classify.
    Saves per-image softmax probability arrays to cfg.probs_path (pickle) for soft-vote ensemble.
    Returns pred_dict[image_id → [(unicode, cx, cy)]] based on this model's argmax alone
    (final ensemble is computed separately by ensemble_soft_vote in rf2).
    ⚠️ CG-2: submission uses center (cx, cy), NOT top-left (x, y).
    """
    import pickle, torch.nn.functional as F

    model = KuzushijiClassifier(model_name=cfg.cls_model).to(cfg.device)
    model.load_state_dict(torch.load(cfg.cls_ckpt, map_location='cpu'))
    model.eval()

    pred_dict  = {}
    probs_dict = {}   # image_id → {'boxes': np.ndarray[N,4], 'probs': np.ndarray[N, NUM_CLASSES]}
    transform  = get_cls_transforms('valid')
    S          = cfg.cls_img_size

    for img_id, boxes in det_results.items():
        img = cv2.imread(os.path.join(img_dir, img_id + '.jpg'))[:, :, ::-1]
        H, W = img.shape[:2]
        preds = []

        if len(boxes) == 0:
            pred_dict[img_id]  = preds
            probs_dict[img_id] = {'boxes': boxes, 'probs': np.zeros((0, NUM_CLASSES), np.float32)}
            continue

        crops = []
        for x0, y0, x1, y1 in boxes:
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            px = int((x1 - x0) * 0.05); py = int((y1 - y0) * 0.05)
            cx0 = max(x0 - px, 0); cy0 = max(y0 - py, 0)
            cx1 = min(x1 + px, W); cy1 = min(y1 + py, H)
            crop = img[cy0:cy1, cx0:cx1]
            if crop.size == 0:
                crop = np.zeros((S, S, 3), dtype=np.uint8)
            crop = transform(image=crop)['image']
            crop = cv2.resize(crop, (S, S)).astype(np.float32) / 255.
            if cfg.cls_grayscale:
                gray = cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                crop = np.stack([gray, gray, gray], axis=2).astype(np.float32) / 255.
            crops.append(crop.transpose(2, 0, 1))

        batch  = torch.tensor(np.stack(crops), dtype=torch.float32).to(cfg.device)
        logits = model(batch)
        probs  = F.softmax(logits, dim=1).cpu().numpy()   # [N, NUM_CLASSES] — save for ensemble
        class_ids = probs.argmax(axis=1)

        for (x0, y0, x1, y1), cls_id in zip(boxes, class_ids):
            unicode_char = class2unicode[int(cls_id)]
            cx = int((x0 + x1) / 2)   # ⚠️ CG-2: center point for submission
            cy = int((y0 + y1) / 2)
            preds.append((unicode_char, cx, cy))

        pred_dict[img_id]  = preds
        probs_dict[img_id] = {'boxes': np.array(boxes, dtype=np.float32), 'probs': probs}

    # ── Save softmax probs to disk for soft-vote ensemble (CG-9) ─────────────
    os.makedirs('./working', exist_ok=True)
    with open(cfg.probs_path, 'wb') as f:
        pickle.dump(probs_dict, f, protocol=4)
    print(f'  Softmax probs saved to {cfg.probs_path} ({len(probs_dict)} images)')

    del model; gc.collect(); torch.cuda.empty_cache()
    return pred_dict


def ensemble_soft_vote(probs_paths, cfg=CFG):
    """
    Soft-vote ensemble: load per-image softmax probs from all runfiles, average, argmax.
    Called only by rf2 (cfg.run_ensemble=True) after all 3 runfiles have finished inference.
    Returns pred_dict[image_id → [(unicode, cx, cy)]].
    """
    import pickle

    # SEQUENTIAL: load all probs dicts
    all_dicts = []
    for p in probs_paths:
        with open(p, 'rb') as f:
            all_dicts.append(pickle.load(f))
        print(f'  Loaded ensemble probs from {p}')

    # Use rf0 as the reference for image_ids and boxes (all runfiles share the same detector)
    ref_dict = all_dicts[0]
    pred_dict = {}

    # LOOP over images
    for img_id, ref_data in ref_dict.items():
        boxes = ref_data['boxes']   # [N, 4] xyxy — same across all runfiles (shared detector)
        if len(boxes) == 0:
            pred_dict[img_id] = []
            continue

        # LOOP: average softmax probs across all models ⚠️ soft-vote > hard-vote for rare classes
        avg_probs = np.zeros_like(ref_data['probs'])   # [N, NUM_CLASSES]
        n_models  = 0
        for d in all_dicts:
            if img_id in d and d[img_id]['probs'].shape == avg_probs.shape:
                avg_probs += d[img_id]['probs']
                n_models  += 1
        if n_models == 0:
            pred_dict[img_id] = []
            continue
        avg_probs /= n_models
        class_ids = avg_probs.argmax(axis=1)

        preds = []
        for (x0, y0, x1, y1), cls_id in zip(boxes, class_ids):
            unicode_char = class2unicode[int(cls_id)]
            cx = int((x0 + x1) / 2)   # ⚠️ CG-2
            cy = int((y0 + y1) / 2)
            preds.append((unicode_char, cx, cy))
        pred_dict[img_id] = preds

    print(f'  Soft-vote ensemble done ({len(pred_dict)} images, {n_models} models averaged)')
    return pred_dict


# ── Evaluation (uses center-inside-bbox, NOT IoU) ⚠️ CG-3 ───────────────────
def evaluate_f1(gt_dict, pred_dict):
    tp = fp = fn = 0
    for img_id, gt_chars in gt_dict.items():
        preds   = pred_dict.get(img_id, [])
        gt_used = [False] * len(gt_chars)
        for pred_ch, pred_cx, pred_cy in preds:
            matched = False
            for gi, (gt_ch, gt_x, gt_y, gt_w, gt_h) in enumerate(gt_chars):
                if gt_used[gi]: continue
                if pred_ch == gt_ch and \
                   gt_x <= pred_cx <= gt_x + gt_w and gt_y <= pred_cy <= gt_y + gt_h:
                    gt_used[gi] = True; matched = True; tp += 1; break
            if not matched: fp += 1
        fn += sum(1 for u in gt_used if not u)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    print(f'F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}')
    return f1, precision, recall


def make_submission(pred_dict, output_path='./submission/submission.csv', cfg=CFG):
    sub = pd.read_csv(cfg.sub_csv)
    sub['labels'] = sub['image_id'].map(
        lambda img_id: ' '.join(
            f'{ch} {cx} {cy}' for ch, cx, cy in pred_dict.get(img_id, [])
        )
    )
    os.makedirs('./submission', exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(f'Saved {output_path}')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    seed_everything(CFG.seed)
    train_df = pd.read_csv(CFG.train_csv, keep_default_na=False)

    # ── Stage 1: Train detector ────────────────────────────────────────────
    if not os.path.exists(CFG.det_ckpt):
        print('=== Stage 1: Training character detector ===')
        train_detector(train_df)
    else:
        print(f'=== Stage 1 skipped — using existing {CFG.det_ckpt} ===')

    # ── Stage 2: Train classifier ──────────────────────────────────────────
    if not os.path.exists(CFG.cls_ckpt):
        print('\n=== Stage 2: Training character classifier ===')
        train_classifier(train_df)
    else:
        print(f'=== Stage 2 skipped — using existing {CFG.cls_ckpt} ===')

    # ── Inference on test images ───────────────────────────────────────────
    print('\n=== Inference: running detector on test images ===')
    sub_df    = pd.read_csv(CFG.sub_csv)
    test_ids  = sub_df['image_id'].tolist()
    det_results = run_detector_on_images(CFG.test_img_dir, test_ids)

    print('=== Inference: classifying detected characters ===')
    pred_dict = classify_crops(CFG.test_img_dir, det_results)

    make_submission(pred_dict)
    print('Done.')


if __name__ == '__main__':
    os.makedirs('./working', exist_ok=True)
    main()
```

---

## Part 4: Trigger Prompts

Now implement the complete **two-stage** solution for the **Kuzushiji Recognition** task.

**Data format** (critical — see CG-1):
- `train.csv`: columns `image_id` and `labels`. The `labels` field contains space-separated groups of **exactly 5 tokens**: `unicode_char x y width height`. Parse in groups of 5 — never treat each token individually.
- `unicode_translation.csv`: the **only** source of truth for class mapping (4787 Unicode strings → integer indices). Load it first and assert `len == 4787`. See CG-4.

**Submission format** (critical — see CG-2):
- Predict the **center** of each detected bbox: `cx = x + w // 2`, `cy = y + h // 2`.
- Each row: `"{unicode} {cx} {cy} {unicode} {cx} {cy} ..."` space-separated.

**Evaluation metric** (critical — see CG-3):
- F1 based on **center-point-inside-bbox**: a prediction matches iff the codepoint matches AND `x ≤ cx ≤ x+w` AND `y ≤ cy ≤ y+h`. **NOT IoU**.

**Stage 1 — Character Detection** (class-agnostic):
- Model: `torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)`, replace head with `FastRCNNPredictor(in_features, 2)` (background + character).
- Train on **512×512 random crops** from full training images (stride sampling to include characters). ⚠️ CG-5 HARD LIMIT.
- Training: SGD lr=0.005, momentum=0.9, **30,000 steps** ⚠️ CG-5 HARD LIMIT.
- Inference: **sliding-window** (stride=256) on full images + **NMS** (iou_thresh=0.3) ⚠️ CG-7 MANDATORY.
- Score threshold: 0.3 at inference.

**Stage 2 — Character Classification** (4787 classes):
- Backbone (choose per runfile — must differ for ensemble diversity): ⚠️ CG-5 HARD LIMIT
  - runfile_0: `tf_efficientnet_b3` (RGB, batch=256, 10 epochs)
  - runfile_1: `tf_efficientnet_b4` (grayscale, batch=128, 10 epochs)
  - runfile_2: `resnet50` (RGB, batch=256, 10 epochs)
- Train on GT crops from `train.csv`: expand bbox 5% each side, resize to **96×96**. ⚠️ CG-5 HARD LIMIT.
- Undersample classes with >2000 occurrences to 2000 to prevent imbalance.
- Augmentation: RandomResizedCrop, brightness/contrast, rotate ±15°, RandomErasing. **NEVER add HorizontalFlip or VerticalFlip**.
- Loss: `nn.CrossEntropyLoss()`. Optimizer: SGD + CosineAnnealingLR.

**Final ensemble**: run all three runfiles; for each detected character box (from shared detector), take **hard vote** (mode) of the three classifier predictions.

Let's think step by step and write your code here.
