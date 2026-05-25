# Blueprint: Denoising Dirty Documents

## Overview

This notebook builds an **image-to-image convolutional autoencoder** using TensorFlow 2 / Keras to remove background noise and stains from scanned document images. It is a complete end-to-end pipeline from data loading through model training to submission generation.

---

## Competition Details

| Field | Value |
|---|---|
| Competition | [Denoising Dirty Documents](https://www.kaggle.com/competitions/denoising-dirty-documents) |
| Task | Image restoration — remove noise/background from scanned document images |
| Input | Noisy greyscale document scans (`train/`, `test/`) |
| Target | Clean versions of the training images (`train_cleaned/`) |
| Evaluation | Mean pixel-level RMSE |

---

## Dependencies

| Library | Role |
|---|---|
| `numpy` | Array math |
| `cv2` (OpenCV) | Image reading, resizing, color conversion |
| `tensorflow.keras` | Autoencoder model, training, callbacks |
| `sklearn.model_selection` | Train/validation split |
| `matplotlib` | Visual comparison of noisy vs. cleaned images |
| `zipfile`, `shutil` | Extracting and cleaning up zipped data |

---

## Data

- **Input:** Zipped archives (`train.zip`, `test.zip`, `train_cleaned.zip`) extracted to `/kaggle/working/`.
- **Train pairs:** Noisy image `train/NNN.png` ↔ clean target `train_cleaned/NNN.png`.
- All images are resized to a **fixed 420×540** resolution before batching.
- Submission format: pixel values flattened and written as `id,value` CSV rows.

---

## Pipeline

### 1. Data Extraction
```python
with zipfile.ZipFile(path_zip + 'train.zip', 'r') as zip_ref:
    zip_ref.extractall(path)
```
Extracts `train/`, `test/`, `train_cleaned/` to the working directory.

### 2. Image Preprocessing (`process_image`)
```python
def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (540, 420))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # convert to single channel
    img /= 255.0                                    # normalize to [0, 1]
    img = img.reshape(420, 540, 1)
    return img
```
All images → float32, 420×540×1 greyscale, normalized.

### 3. Train/Validation Split
```python
X_train, X_val, Y_train, Y_val = train_test_split(
    np.asarray(train), np.asarray(train_cleaned),
    test_size=0.15, random_state=42
)
```
85% training / 15% validation.

### 4. Model Architecture (Convolutional Autoencoder)

```
Input: (420, 540, 1)

Encoder:
  Conv2D(64, 3×3, relu, same) → MaxPool2D(2×2)  →  (210, 270, 64)

Decoder:
  Conv2D(64, 3×3, relu, same) → UpSampling2D(2×2)  →  (420, 540, 64)
  Conv2D(1,  5×5, sigmoid, same)                      →  (420, 540, 1)
```

Bottleneck: `210×270×64` (halved spatial resolution, 64 filters). The encoder compresses noisy input; the decoder reconstructs a clean image.

```python
input_layer = Input(shape=(420, 540, 1))
x = Conv2D(64, (3,3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
output_layer = Conv2D(1, (5,5), activation='sigmoid', padding='same')(x)
model = Model(input_layer, output_layer)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
```

### 5. Training
```python
callback = EarlyStopping(monitor='loss', patience=30)
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=600, batch_size=24,
    verbose=0, callbacks=[callback]
)
```

- **Optimizer:** Adam
- **Loss:** MSE (pixel-level)
- **Metric:** MAE
- **Early stopping:** patience=30 on training loss
- **Max epochs:** 600; typical convergence around 100–200 epochs
- Achieves `val_loss < 0.0004` with this architecture

### 6. Inference
```python
Y_test = model.predict(X_test, batch_size=16)
```

### 7. Submission Generation
Each test image is read at its original resolution, denoised by the model, and the pixel values are written row by row:
```python
for i, f in enumerate(test_img):
    imgid = int(f[:-4])
    img = cv2.imread(file, 0)
    img_shape = img.shape
    img = process_image(file).reshape(1, 420, 540, 1)
    pred = model.predict(img)
    pred = cv2.resize(pred.reshape(420, 540), (img_shape[1], img_shape[0]))
    for r in range(img_shape[0]):
        for c in range(img_shape[1]):
            ids.append(str(imgid) + '_' + str(r+1) + 'x' + str(c+1))
            vals.append(pred[r][c])
```

---

## Output

| File | Description |
|---|---|
| `submission.csv` | `id` (image_row×col) + `value` (predicted clean pixel intensity) |

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| Fixed resize to 420×540 | Enables batching of variable-size images |
| Greyscale conversion | Documents are monochrome; discards irrelevant color info |
| Symmetric encoder-decoder | Simple U-Net-style reconstruction without skip connections |
| Sigmoid output | Constrains output to `[0, 1]` matching normalized target pixels |
| MSE loss | Directly optimizes pixel-level reconstruction error |
| Patience=30 EarlyStopping | Allows long plateau phases typical of autoencoders |

---

## Suggested Improvements

- Add skip connections between encoder and decoder layers (full U-Net architecture).
- Use more encoder stages (3–4 pooling levels) for better feature abstraction.
- Add BatchNormalization and Dropout for regularization.
- Apply data augmentation (horizontal flip, brightness jitter) to increase effective dataset size.
- Train at original resolution with patch-based batching instead of resizing.
