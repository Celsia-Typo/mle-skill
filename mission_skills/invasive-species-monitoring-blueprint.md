# Blueprint: Invasive Species Monitoring

## Overview

This notebook is a complete **binary image classification pipeline** using Keras with a pretrained VGG16 backbone to detect the presence of invasive hydrangea plants in forest images. It covers image preprocessing, model construction via transfer learning, training with data augmentation, and submission generation.

---

## Competition Details

| Field | Value |
|---|---|
| Competition | [Invasive Species Monitoring](https://www.kaggle.com/competitions/invasive-species-monitoring) |
| Task | Binary classification — does the image contain invasive hydrangea? |
| Input | High-resolution forest photographs (`.jpg`) |
| Target | `invasive` = 1 (present) or 0 (absent) |
| Evaluation | AUC (Area Under the ROC Curve) |

---

## Dependencies

| Library | Role |
|---|---|
| `keras` (`tensorflow.keras`) | VGG16, model building, training |
| `cv2` (OpenCV) | Image reading, resizing, color conversion |
| `numpy`, `pandas` | Array math and submission handling |
| `matplotlib` | Visualization |
| `glob`, `os` | File discovery |

---

## Data

- **`train_labels.csv`:** columns `name` (integer ID) and `invasive` (0 or 1)
- **`train/`:** ~2,295 training images as `{name}.jpg`
- **`test/`:** ~1,531 test images as `{name}.jpg`
- **`sample_submission.csv`:** `name` column, `invasive` column to fill

---

## Pipeline

### 1. Image Preprocessing (`centering_image`)

All images are resized to fit within 256×256 while preserving aspect ratio, then **centre-cropped to 224×224**:

```python
def centering_image(img):
    size = [256, 256]
    img_size = img.shape[:2]
    row = (size[1] - img_size[0]) // 2   # vertical padding
    col = (size[0] - img_size[1]) // 2   # horizontal padding
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img
    return resized

# Resize to fit within 256×256
if img.shape[0] > img.shape[1]:
    tile_size = (int(img.shape[1]*256/img.shape[0]), 256)
else:
    tile_size = (256, int(img.shape[0]*256/img.shape[1]))
img = centering_image(cv2.resize(img, dsize=tile_size))
img = img[16:240, 16:240]   # centre-crop: 256 → 224
```

Images are read in BGR (OpenCV default) and converted to **RGB** before processing.

### 2. Train/Validation Split

```python
random_index = np.random.permutation(data_num)   # random shuffle
val_split_num = int(round(0.2 * len(y)))          # 20% validation
x_train, y_train = x[val_split_num:], y[val_split_num:]
x_val,   y_val   = x[:val_split_num], y[:val_split_num]

x_train = x_train.astype('float32') / 255.0
x_val   = x_val.astype('float32')   / 255.0
```

Simple random 80/20 split (not stratified). Pixel values normalized to `[0, 1]`.

### 3. Model Architecture — VGG16 + Custom Head

```python
base_model = applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

Custom classification head:
```python
add_model = Sequential([
    Flatten(input_shape=base_model.output_shape[1:]),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')    # binary output
])
model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
```

**VGG16 weights are frozen** (no `base_model.trainable = False` call, but `weights='imagenet'` and no fine-tuning loop means only the head is trained).

### 4. Compilation

```python
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy']
)
```

SGD with momentum is used rather than Adam — a standard choice for fine-tuning pretrained CNNs as it tends to generalize better.

### 5. Training with Augmentation

```python
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=32),
    steps_per_epoch=x_train.shape[0] // 32,
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[ModelCheckpoint('VGG16-transferlearning.model',
                               monitor='val_acc', save_best_only=True)]
)
```

Augmentations: ±30° rotation, ±10% horizontal/vertical shift, horizontal flip. Best model saved by validation accuracy via `ModelCheckpoint`.

### 6. Inference & Submission

```python
test_images = test_images.astype('float32') / 255.0
predictions = model.predict(test_images)

for i, name in enumerate(test_names):
    sample_submission.loc[sample_submission['name'] == name, 'invasive'] = predictions[i]
sample_submission.to_csv("submit.csv", index=False)
```

Raw sigmoid probabilities are written directly to the submission (AUC metric does not require thresholding).

---

## Output

| File | Description |
|---|---|
| `VGG16-transferlearning.model` | Best checkpoint by `val_acc` |
| `submit.csv` | `name` + `invasive` (probability score) |

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| VGG16 pretrained on ImageNet | Strong low-level visual features (edges, textures) transfer well to plant detection |
| Centre-crop to 224×224 | Matches VGG16's expected input size; preserves aspect ratio via padding before crop |
| Aspect-ratio preserving resize | Avoids distorting plant shape features important for species identification |
| SGD + momentum (lr=1e-4) | Low learning rate for fine-tuning; SGD generalizes better than Adam on small datasets |
| Sigmoid + binary_crossentropy | Binary task; AUC evaluation does not require argmax thresholding |
| Augmentation (rotation, shift, flip) | ~2,300 training images is small; augmentation is critical to prevent overfitting |

---

## Suggested Improvements

- Unfreeze the last few VGG16 blocks and fine-tune with a very low learning rate (e.g., 1e-5).
- Replace VGG16 with EfficientNet-B3 or ResNet50 for better accuracy/parameter trade-off.
- Use stratified split to ensure balanced class ratio in train and validation.
- Add more aggressive augmentations: zoom, brightness/contrast jitter, Gaussian blur.
- Monitor `val_auc` rather than `val_acc` for model selection (matches the competition metric).
- Use `model.fit()` instead of deprecated `model.fit_generator()` (TF 2.x).
