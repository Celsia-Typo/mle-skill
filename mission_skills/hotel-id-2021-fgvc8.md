# hotel-id-2021-fgvc8

## Overview

This notebook is a baseline solution for the **Hotel-ID 2021 (FGVC8)** Kaggle competition. The goal is to classify hotel images to specific hotel IDs — a task with direct applications in combating human trafficking by identifying hotel locations from images.

---

## Purpose

Train a convolutional neural network (ResNet34) to classify images into one of ~7,770 hotel categories, then generate a Kaggle-style `submission.csv`.

---

## Dependencies

| Library | Role |
|---|---|
| `numpy`, `pandas` | Data manipulation |
| `PIL` (Pillow) | Image loading |
| `torch`, `torchvision` | Model training (PyTorch) |
| `sklearn.preprocessing` | Label encoding |

---

## Configuration / Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `BATCH` | 16 | Training batch size |
| `EPOCHS` | 3 | Number of training epochs |
| `LR` | 0.001 | Learning rate (Adam) |
| `IM_SIZE` | 128 | Image resize target (pixels) |
| `DEVICE` | cuda / cpu | Auto-detected GPU or CPU |

---

## Data

- **Source:** `../input/hotel-id-2021-fgvc8/`
- **Training images:** `train_images/` (organised by hotel chain subdirectory)
- **Test images:** `test_images/`
- **Labels CSV:** `train.csv` — columns include `hotel_id`, `chain`, `image`

---

## Pipeline

### 1. Data Loading & Exploration
- Load `train.csv` with pandas.
- Explore hotel ID and chain distribution (value counts, filtering).

### 2. Label Encoding
- Use `sklearn.LabelEncoder` to convert `hotel_id` strings → integer labels.
- Store a reverse mapping (`class_map`) for inference decoding.

### 3. Image Path Construction
- Build a `chain_image` column combining chain subdirectory and filename: `"{chain}/{image}"`.
- Extract `X_Train` (image paths) and `Y_Train` (integer labels) arrays.

### 4. Transforms
```
Compose([ToTensor(), Resize(128, 128), Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
```

### 5. Custom Dataset (`GetData`)
- Accepts: directory path, filenames, labels, transforms.
- Returns `(transformed_image, label)` for training; `(transformed_image, filename)` for testing.

### 6. DataLoader
- `trainloader`: batch_size=16, shuffle=True.
- `testloader`: batch_size=1, shuffle=False.

### 7. Model
- Base: `torchvision.models.resnet34()` (pretrained weights not explicitly loaded).
- Final FC layer replaced: `nn.Linear(512, NUM_CL)` where `NUM_CL = 7770`.
- Moved to `DEVICE`.

### 8. Training Loop
- Loss: `nn.CrossEntropyLoss`
- Optimizer: `torch.optim.Adam(lr=0.001)`
- 3 epochs; logs `Epoch | Loss` per epoch.

### 9. Inference
- Collect test filenames from `TEST_DIR`.
- Forward pass with `torch.no_grad()`; apply softmax via `torch.exp(logits)`.
- Take top-1 predicted class index.

### 10. Submission
- Decode predicted integer labels back to `hotel_id` via `class_map`.
- Save `submission.csv` with columns `['image', 'hotel_id']`.

---

## Output

| File | Description |
|---|---|
| `submission.csv` | Final predictions: `image` → `hotel_id` |

---

## Key Design Choices

- Images are resized uniformly to 128×128 for speed; increasing this improves accuracy.
- The dataset class handles both train (returns labels) and test (returns filenames) modes via directory name check.
- A commented-out block limits training to 1,000 samples for fast prototyping.
- No validation split is implemented — this is a pure baseline.

---

## Suggested Improvements

- Add a train/validation split to monitor overfitting.
- Use a pretrained ResNet (`pretrained=True`) for better transfer learning.
- Increase `IM_SIZE` (e.g., 224 or 384) for higher accuracy.
- Add metric logging (top-5 accuracy is standard for this competition).
- Use learning rate scheduling (cosine annealing, etc.).
