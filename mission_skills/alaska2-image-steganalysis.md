# Blueprint: ALASKA2 Image Steganalysis

> ⚠️ **Nature of this notebook: Training-defined, inference-executed pipeline.**
> The training code (`Fitter`, `run_training()`, `BalanceClassSampler`) is fully written but `run_training()` is commented out. The notebook loads a pretrained checkpoint from `alaska2-public-baseline` for inference and generates `submission.csv`. Running the uncommented `run_training()` produces a fully trained model.
> Source file: `alaska2-image-steganalysis.ipynb`

---

## Competition Details

| Field | Value |
|---|---|
| Competition | [ALASKA2 Image Steganalysis](https://www.kaggle.com/competitions/alaska2-image-steganalysis) |
| Task | 4-class classification: detect which steganographic algorithm was applied to a JPEG image (or none) |
| Input | JPEG images (512×512 typical) |
| Classes | `Cover` (0) — clean; `JMiPOD` (1), `JUNIWARD` (2), `UERD` (3) — stego algorithms |
| Evaluation | Weighted AUC: 2× weight on TPR ∈ [0, 0.4], 1× weight on TPR ∈ [0.4, 1.0] |

---

## Dependencies

| Library | Role |
|---|---|
| `efficientnet_pytorch` | EfficientNet-B2 backbone (`pip install -q efficientnet_pytorch`) |
| `albumentations`, `ToTensorV2` | Image augmentation pipeline |
| `torch`, `torch.nn` | Model building, training loop |
| `catalyst.data.sampler.BalanceClassSampler` | Class-balanced sampling during training |
| `cv2` | Image loading (BGR→RGB conversion) |
| `sklearn.model_selection.GroupKFold` | Fold splitting grouped by image filename |
| `sklearn.metrics.roc_curve`, `auc` | Weighted AUC metric computation |
| `pandas`, `numpy` | Data manipulation |

**GPU required:** `net.cuda()` and `torch.device('cuda:0')` — training and inference are hard-coded to CUDA.

---

## Data

| File / Directory | Description |
|---|---|
| `alaska2-image-steganalysis/{Cover,JMiPOD,JUNIWARD,UERD}/*.jpg` | Training images organized by class folder |
| `alaska2-image-steganalysis/Test/*.jpg` | Test images (unlabeled) |
| `alaska2-public-baseline/best-checkpoint-033epoch.bin` | Pretrained EfficientNet-B2 checkpoint (epoch 33) |
| `alaska2-public-baseline/log.txt` | Training log from the pretrained run |

**Data structure note:** Each base JPEG image exists in all four class folders with the same filename — the same cover image has a corresponding JMiPOD, JUNIWARD, and UERD stego version. This is why GroupKFold by `image_name` is essential.

---

## Pipeline

### Step 1 — Dataset Construction and GroupKFold Splitting

```python
dataset = []
for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
    for path in glob('../input/alaska2-image-steganalysis/Cover/*.jpg'):
        dataset.append({
            'kind': kind,
            'image_name': path.split('/')[-1],
            'label': label
        })

random.shuffle(dataset)
dataset = pd.DataFrame(dataset)

gkf = GroupKFold(n_splits=5)
dataset.loc[:, 'fold'] = 0
for fold_number, (train_index, val_index) in enumerate(
        gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):
    dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number
```

- The `glob` always reads filenames from `Cover/` — this is intentional because all classes share the same base image names; the `kind` variable controls the actual folder path at load time
- **GroupKFold by `image_name`** ensures the same base image (and its stego counterparts) never appear in both train and validation — critical for preventing data leakage since identical images exist across all 4 class folders
- 5 folds; notebook uses `fold_number = 0`

### Step 2 — Augmentations

```python
def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height=512, width=512, p=1.0),
        ToTensorV2(p=1.0),
    ])

def get_valid_transforms():
    return A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        ToTensorV2(p=1.0),
    ])
```

- Only flips and resize — no color jitter, no normalization (images are divided by 255.0 in `__getitem__` but no ImageNet mean/std subtraction)
- `ToTensorV2`: converts HWC NumPy arrays to CHW tensors without rescaling

### Step 3 — Dataset and One-Hot Targets

```python
class DatasetRetriever(Dataset):
    def __getitem__(self, index):
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        image = cv2.imread(f'{DATA_ROOT_PATH}/{kind}/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = self.transforms(image=image)
            image = sample['image']
        target = onehot(4, label)   # one-hot vector of size 4
        return image, target
```

- Images loaded via cv2 (BGR) then converted to RGB
- Pixel values normalized to [0, 1] by dividing by 255 (no mean/std normalization)
- Targets returned as one-hot float32 vectors of size 4

### Step 4 — Evaluation Metric: Weighted AUC

```python
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]  # 2× weight on low-FPR region, 1× on high-FPR region

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    normalization = np.dot(areas, weights)   # = 0.4*2 + 0.6*1 = 1.4

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min, y_max = tpr_thresholds[idx], tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        x = np.concatenate([fpr[mask], np.linspace(fpr[mask][-1], 1, 100)])
        y = np.concatenate([tpr[mask], [y_max] * 100]) - y_min
        competition_metric += metrics.auc(x, y) * weight

    return competition_metric / normalization
```

- **Binary score derivation:** `y_pred = 1 - softmax(output)[:,0]` — probability of NOT being Cover class; `y_true = argmax(onehot_target).clip(0, 1)` — 0 for Cover, 1 for any stego
- The 4-class problem is converted to binary (Cover vs. any stego) for this metric
- The metric emphasizes the low-FPR operating region (TPR 0–0.4 is twice as important), reflecting the real-world cost of alerting on clean images

### Step 5 — Label Smoothing Loss

```python
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.confidence = 1.0 - smoothing   # 0.95
        self.smoothing = smoothing           # 0.05

    def forward(self, x, target):
        if self.training:
            logprobs = F.log_softmax(x, dim=-1)
            nll_loss = -(logprobs * target).sum(-1)          # cross-entropy with soft target
            smooth_loss = -logprobs.mean(dim=-1)             # uniform distribution term
            return (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()
        else:
            return F.cross_entropy(x, target)
```

- Soft target: `0.95 × true_class_NLL + 0.05 × uniform_NLL`
- In eval mode: falls back to standard `F.cross_entropy` — target is one-hot (integer-compatible)
- Prevents the model from becoming overconfident on subtle steganographic differences

### Step 6 — Model: EfficientNet-B2

```python
from efficientnet_pytorch import EfficientNet

def get_net():
    net = EfficientNet.from_pretrained('efficientnet-b2')
    net._fc = nn.Linear(in_features=1408, out_features=4, bias=True)
    return net

net = get_net().cuda()
```

- `from_pretrained('efficientnet-b2')`: downloads ImageNet pretrained weights
- Replaces the final fully-connected layer: 1408-dim → 4-class
- All layers remain trainable (no freeze)

### Step 7 — Fitter: Training Loop with Checkpoint Management

```python
class Fitter:
    def __init__(self, model, device, config):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        # Note: weight decay is configured but AdamW is called with model.parameters() directly
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                           patience=1, threshold=0.0001, min_lr=1e-8)
        self.criterion = LabelSmoothing().to(self.device)

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            self.train_one_epoch(train_loader)
            self.save(f'{self.base_dir}/last-checkpoint.bin')   # always save last

            summary_loss, final_scores = self.validation(validation_loader)
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                # keep only the 3 most recent best checkpoints
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            self.scheduler.step(metrics=summary_loss.avg)   # validation_scheduler=True
```

**Checkpoint naming:** `best-checkpoint-033epoch.bin` — zero-padded 3-digit epoch number. The pretrained model was saved at epoch 33 of 25 configured epochs (training was likely resumed or ran longer).

### Step 8 — Class-Balanced DataLoader

```python
from catalyst.data.sampler import BalanceClassSampler

train_loader = DataLoader(
    train_dataset,
    sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
    batch_size=16,
    drop_last=True,
    num_workers=4,
)
```

- `mode="downsampling"`: all 4 classes downsampled to the size of the smallest class
- Ensures equal class representation per epoch — Cover images (which are more numerous) are subsampled to match each stego class
- `drop_last=True`: avoids incomplete batches that could bias the last gradient step

### Step 9 — Inference with Pretrained Checkpoint

```python
checkpoint = torch.load('../input/alaska2-public-baseline/best-checkpoint-033epoch.bin')
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

result = {'Id': [], 'Label': []}
for step, (image_names, images) in enumerate(data_loader):
    y_pred = net(images.cuda())
    y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]
    result['Id'].extend(image_names)
    result['Label'].extend(y_pred)

submission = pd.DataFrame(result)
submission.to_csv('submission.csv', index=False)
```

- `1 - softmax[:, 0]`: inverts the Cover probability → high score means "likely steganographic"
- Inference batch size = 8 (small; the model is large at 512×512 input)
- `DatasetSubmissionRetriever` applies `get_valid_transforms()` (resize only, no flips)

---

## Output

| File | Description |
|---|---|
| `submission.csv` | `Id` (filename), `Label` (float in [0, 1] — probability of steganography) |
| `best-checkpoint-{epoch:03d}epoch.bin` | Saved checkpoints (model, optimizer, scheduler, epoch, best loss) |
| `last-checkpoint.bin` | Latest epoch checkpoint for training resumption |
| `log.txt` | Training log with epoch-by-epoch loss and weighted AUC |

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| GroupKFold by `image_name` | Prevents leakage: the same base image appears in all 4 class folders; without grouping, the model would memorize image identity rather than stego signals |
| `BalanceClassSampler(mode="downsampling")` | Cover class is much larger than the 3 stego classes; downsampling prevents the model from biasing toward Cover at the cost of stego detection |
| `1 - softmax[:, 0]` binary score | The competition evaluates Cover vs. any stego (binary AUC), not 4-class accuracy; this converts the 4-class logits to a single steganography probability |
| Weighted AUC (2× on TPR 0–0.4) | Low FPR is more valuable in practice; false positives (flagging clean images) are more costly than missing some stego images; competition metric reflects this asymmetry |
| Label smoothing (0.05) | Steganographic differences are subtle; overconfident predictions can be brittle; smoothing regularizes the model to output calibrated probabilities |
| ReduceLROnPlateau(patience=1) | Aggressive LR reduction — halves LR if validation loss does not improve for a single epoch; suitable for training long enough (25 epochs) where fine-grained LR annealing helps |
| Checkpoint retention (top 3) | Prevents disk overflow while keeping fallback checkpoints in case the last best was a fluke |
| No ImageNet normalization | The notebook omits mean/std normalization; this is a limitation — EfficientNet pretrained weights expect normalized inputs (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |

---

## SOTA Gap

| Aspect | This Notebook | Competition SOTA |
|---|---|---|
| Backbone | EfficientNet-B2 (1408-dim head) | EfficientNet-B7, EfficientNet-B6, ViT ensembles |
| Image size | 512×512 | 512×512 to full resolution (steganography is pixel-level) |
| Augmentation | HFlip + VFlip only | Rich stego-aware augmentations: JPEG re-compression, DCT-domain augmentations |
| TTA | None | Test-time augmentation (4–8 flips/rotations averaged) |
| Folds used | 1 fold (fold 0) | 5-fold cross-validation, all folds ensembled |
| Models ensembled | 1 model | 3–5 diverse architectures (EfficientNet-B5/B6/B7 + SRM-filtered models) |
| Input preprocessing | Pixel normalization only | SRM (Spatial Rich Model) filter residuals as additional input channels |
| Normalization | Missing ImageNet mean/std | Proper ImageNet normalization before pretrained backbone |
| Public LB | ~0.920 (referenced) | ~0.940 (top solutions) |

---

## Suggested Improvements

1. **Add ImageNet mean/std normalization** — EfficientNet-B2 was pretrained with `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`; omitting this shifts the input distribution and degrades pretrained feature quality
2. **Add Test-Time Augmentation (TTA)** — averaging predictions over horizontal flip, vertical flip, and their combination (4 passes) is straightforward and typically gains 0.002–0.005 weighted AUC
3. **Train all 5 folds and ensemble** — using only fold 0 leaves ~80% of training data unused for the final submission; training all 5 folds and averaging their predictions is the single largest improvement available
4. **Add SRM (Spatial Rich Model) filter channels** — prepend the 3 SRM high-pass filter residuals as extra input channels; top solutions show significant gains because SRM explicitly highlights steganographic noise patterns invisible to standard CNNs
5. **Scale up to EfficientNet-B5 or B6** — larger EfficientNet variants have more capacity to model the subtle pixel-level artifacts introduced by steganographic algorithms; B5 trained on the full 512×512 is a common baseline for top-10 solutions
6. **Use full resolution (no resize)** — steganographic modifications are pixel-level; resizing to 512×512 may remove or distort the exact DCT coefficients that were altered; using original resolution (or padding instead of resizing) preserves all signal
7. **Add richer augmentations cautiously** — JPEG re-compression at various quality factors (as an augmentation) helps generalize to test images re-compressed at different qualities; however, aggressive spatial augmentations can destroy steganographic signals, so augmentation choices must be stego-aware
8. **Use AdamW with proper weight decay grouping** — the `Fitter` constructs `optimizer_grouped_parameters` separating bias/LayerNorm weights but then passes `self.model.parameters()` directly to `AdamW`, discarding the grouping; fixing this ensures bias/norm params are not weight-decayed
