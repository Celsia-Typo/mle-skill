# Blueprint: Plant Seedlings Classification

> ✅ **Nature of this notebook: Full end-to-end pipeline.**
> Covers data loading, subsampling, Xception bottleneck feature extraction (no fine-tuning), Logistic Regression classifier training, validation evaluation, confusion matrix, and `submission.csv` generation.
> Source file: `plant-seedlings-classification.ipynb`

---

## Competition Details

| Field | Value |
|---|---|
| Competition | [Plant Seedlings Classification](https://www.kaggle.com/competitions/plant-seedlings-classification) |
| Task | Multi-class classification of plant seedling species from images |
| Input | RGB images of plant seedlings at various growth stages |
| Target | One of 12 species labels |
| Evaluation | Mean F1 score (categorization accuracy) |

---

## Dependencies

| Library | Role |
|---|---|
| `keras.applications.xception` | Xception pretrained model for bottleneck feature extraction |
| `keras.preprocessing.image` | Image loading and array conversion |
| `sklearn.linear_model.LogisticRegression` | Classifier trained on extracted features |
| `sklearn.metrics` | `accuracy_score`, `confusion_matrix` |
| `numpy`, `pandas` | Array manipulation and tabular data |
| `matplotlib`, `seaborn` | Image grid visualization and confusion matrix heatmap |
| `tqdm` | Progress bars during feature extraction |

**Offline model loading (Kaggle environment without internet):**
```bash
cp ../input/keras-pretrained-models/xception* ~/.keras/models/
```
Xception weights are copied from a local dataset to the Keras cache directory before loading.

---

## Data

| Path | Description |
|---|---|
| `train/{species}/` | 12 subdirectories, one per species class, containing training images |
| `test/` | Flat directory of test images (no labels) |
| `sample_submission.csv` | `file`, `species` columns |

**12 Species Classes:**
Black-grass, Charlock, Cleavers, Common Chickweed, Common Wheat, Fat Hen, Loose Silky-bent, Maize, Scentless Mayweed, Shepherd's Purse, Small-flowered Cranesbill, Sugar Beet

**Class sizes (full dataset):** vary per species; per-category counts printed at runtime.

---

## Pipeline

### Step 1 — Build Train DataFrame

```python
train = []
for category_id, category in enumerate(CATEGORIES):
    for file in os.listdir(os.path.join(train_dir, category)):
        train.append(['train/{}/{}'.format(category, file), category_id, category])
train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])
```

### Step 2 — Subsample (200 images per class)

```python
SAMPLE_PER_CATEGORY = 200
train = pd.concat([train[train['category'] == c][:SAMPLE_PER_CATEGORY] for c in CATEGORIES])
train = train.sample(frac=1)   # shuffle
train.index = np.arange(len(train))
# Total: 12 × 200 = 2,400 training images used
```

- Subsampling is necessary because CPU-only Kaggle kernels cannot process the full dataset within the time limit
- Images are taken in filesystem order (first 200 per class) — no stratification within the sample

### Step 3 — Train / Validation Split (80/20)

```python
SEED = 1987
np.random.seed(seed=SEED)
rnd = np.random.random(len(train))
train_idx = rnd < 0.8    # ~1,920 images
valid_idx = rnd >= 0.8   # ~480 images
ytr = train.loc[train_idx, 'category_id'].values
yv  = train.loc[valid_idx, 'category_id'].values
```

- Random holdout split (not stratified — class balance not guaranteed per fold)

### Step 4 — Image Loading and Preprocessing

```python
INPUT_SIZE = 299   # Xception native input resolution

def read_img(filepath, size):
    img = image.load_img(os.path.join(data_dir, filepath), target_size=size)
    img = image.img_to_array(img)
    return img

x_train = np.zeros((len(train), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
for i, file in tqdm(enumerate(train['file'])):
    img = read_img(file, (INPUT_SIZE, INPUT_SIZE))
    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    x_train[i] = x
```

- `xception.preprocess_input`: scales pixel values to [-1, 1] (Xception-specific normalization)
- All images resized to 299×299 to match Xception's expected input
- Full train array held in memory as float32: shape `(2400, 299, 299, 3)`

### Step 5 — Xception Bottleneck Feature Extraction

```python
POOLING = 'avg'
xception_bottleneck = xception.Xception(
    weights='imagenet',
    include_top=False,    # remove classification head
    pooling=POOLING       # GlobalAveragePooling2D after last conv block
)

train_x_bf = xception_bottleneck.predict(Xtr, batch_size=32, verbose=1)
valid_x_bf = xception_bottleneck.predict(Xv,  batch_size=32, verbose=1)
# Output shape: (n_samples, 2048) — 2048-dim feature vector per image
```

- `include_top=False` + `pooling='avg'`: applies GlobalAveragePooling2D to the final conv layer output, producing a 2048-dim vector per image
- Weights frozen at ImageNet values — no fine-tuning performed
- Feature extraction runs on CPU (no GPU available in original Kaggle kernel)

### Step 6 — Logistic Regression Classifier

```python
logreg = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    random_state=SEED
)
logreg.fit(train_x_bf, ytr)
valid_preds = logreg.predict(valid_x_bf)
valid_probs = logreg.predict_proba(valid_x_bf)
```

- Multinomial logistic regression with L-BFGS solver — appropriate for 12-class softmax output
- Default regularization: L2, `C=1.0`
- Trains directly on 2048-dim Xception features (no additional hidden layers)

### Step 7 — Validation Evaluation

```python
print('Validation Xception Accuracy {}'.format(accuracy_score(yv, valid_preds)))
```

- Confusion matrix plotted as seaborn heatmap with abbreviated class labels
- Saved to `Confusion matrix.png` at 300 DPI

### Step 8 — Test Inference and Submission

```python
x_test = np.zeros((len(test), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
for i, filepath in tqdm(enumerate(test['filepath'])):
    img = read_img(filepath, (INPUT_SIZE, INPUT_SIZE))
    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    x_test[i] = x

test_x_bf  = xception_bottleneck.predict(x_test, batch_size=32, verbose=1)
test_preds = logreg.predict(test_x_bf)

test['category_id'] = test_preds
test['species'] = [CATEGORIES[c] for c in test_preds]
test[['file', 'species']].to_csv('submission.csv', index=False)
```

---

## Output

| File | Description |
|---|---|
| `submission.csv` | `file`, `species` — one predicted class label per test image |
| `Confusion matrix.png` | 12×12 heatmap of validation predictions vs. ground truth |

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| Xception as feature extractor (frozen) | No GPU available in original Kaggle CPU kernel; full fine-tuning is infeasible — bottleneck extraction + shallow classifier is the CPU-viable approach |
| `include_top=False, pooling='avg'` | Removes Xception's 1000-class ImageNet head; GlobalAveragePooling produces a fixed 2048-dim vector regardless of spatial resolution |
| Subsample to 200 per class | Full dataset processing on CPU would exceed the Kaggle kernel time limit (~1 hour); 2400 images complete within the limit |
| Logistic Regression (not MLP) | Fast to train on 2048-dim features with 2400 samples; interpretable; avoids overfitting on the small subsample |
| `solver='lbfgs'` with `multi_class='multinomial'` | lbfgs handles multinomial loss directly and converges well on moderate-sized feature matrices |
| INPUT_SIZE = 299 | Xception's native resolution — using smaller sizes would require spatial interpolation and degrade feature quality |
| 80/20 random split (not KFold) | Simple holdout sufficient given CPU constraints; KFold would multiply feature extraction time by K |

---

## SOTA Gap

| Aspect | This Notebook | Competition SOTA |
|---|---|---|
| Model | Xception features + LogReg (no fine-tuning) | Fine-tuned EfficientNet-B4/B7, ResNet-50, Inception-ResNet-V2 |
| Training data | 2,400 images (200/class subsample) | Full dataset (~4,750 images) |
| Augmentation | None | Random flips, rotations, color jitter, cutout |
| Classifier head | Logistic Regression | Fine-tuned dense layers with dropout |
| Validation accuracy | ~90–93% (reported as near 1-hour limit) | ~98–99% (top LB) |
| Evaluation | Holdout accuracy | Stratified KFold F1 |
| TTA | None | Multi-crop / flip TTA |

---

## Suggested Improvements

1. **Fine-tune Xception end-to-end** — unfreeze the top conv blocks and train with a low learning rate (1e-5); fine-tuning on the full dataset closes the ~5–8% gap to SOTA on this competition
2. **Use full training data** — the 200/class cap discards ~50% of available images; even on CPU, a smaller model (MobileNetV2) trained on the full set would outperform Xception features on a subsample
3. **Add data augmentation** — `ImageDataGenerator` with horizontal/vertical flips, random rotations (±15°), and zoom (0.8–1.2×) significantly improves generalization on plant images where orientation is arbitrary
4. **Replace LogReg with a small MLP** — a 2-layer MLP (Dense 512 → Dropout 0.5 → Dense 12 → Softmax) on Xception features typically gains 2–3% accuracy over LogReg
5. **Use stratified KFold** — the current random 80/20 split is not stratified; with only 200 samples per class, random splits can produce imbalanced folds; `StratifiedKFold` ensures equal class representation
6. **Add test-time augmentation (TTA)** — average predictions over horizontally flipped and/or slightly rotated versions of each test image to reduce prediction variance
7. **Try EfficientNetB3/B4** — EfficientNet typically outperforms Xception at similar parameter counts; `efficientnet` package provides Keras-compatible pretrained models with the same bottleneck extraction approach
