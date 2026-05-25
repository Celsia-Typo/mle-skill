# APTOS 2019 Blindness Detection — Blueprint

## 1. Problem Understanding
- **Task:** Single-label ordinal classification (fundus retinal photography).
- **Metric:** Quadratic Weighted Kappa (QWK).
- **Target:** 5-class DR severity — `0=No DR`, `1=Mild`, `2=Moderate`, `3=Severe`, `4=Proliferative`.
- **Key Challenges:** Class imbalance (0 & 2 dominate), variable lighting, large uninformative black borders, subtle lesions (microaneurysms, cotton wool spots), inter-grader label noise.

---

## 2. Data Pipeline
- **`load_data()`**: Parse `train/test.csv`, map `id_code` → `.png` path, shuffle with `SEED=77`.
- **`preprocess()`**: Constants: `IMG_SIZE=512`, `NUM_CLASSES=5`, `CHANNEL=3`.
  - `crop_image_from_gray(img, tol=7)`: Grayscale threshold mask → crop all 3 channels via `np.ix_()` simultaneously. **Guard:** return original if crop result is zero-dimension.
  - Ben Graham normalization: `cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)` with `sigmaX=10`.
  - Resize to `IMG_SIZE` **after** cropping.
- **`load_ben_color(path, sigmaX=10)`** ← primary loader: `Read → BGR→RGB → crop → resize → addWeighted`.
- **`circle_crop(path, sigmaX=10)`** ← optional: circular bitwise mask → re-crop → addWeighted. May clip peripheral lesions.

---

## 3. Model Design
- **Backbone:** ResNet-50 (original); recommended: EfficientNet-B4/B5.
- **Input:** `512×512` RGB. Do not reduce resolution.
- **Head:** Ordinal regression (single logit + threshold rounding) or cumulative label encoding.

---

## 4. Training Strategy
- **Loop:** PyTorch AMP training loop.
- **Loss:** `BCEWithLogitsLoss` (ordinal encoding) or `MSELoss` (regression head).
- **Optimizer:** AdamW.
- **Augmentations:** `RandomContrast`, `RandomBrightness`, `GaussNoise`, `ShiftScaleRotate`. Avoid heavy color jitter — Ben Graham already normalizes lighting.
- **Imbalance:** `sklearn.utils.class_weight` to upweight rare classes (1, 3, 4).

---

## 5. Validation Strategy
- **`create_folds()`**: `StratifiedKFold(n_splits=5)` on 5-class label.
- **`validate()`**: Primary metric = QWK. Store OOF predictions for threshold tuning.

---

## 6. Inference Pipeline
- **`predict()`**: 5-fold ensemble. Apply identical `load_ben_color(sigmaX=10)` preprocessing.
- **TTA:** Horizontal flip + multi-scale (448 / 512 / 576), average logits.
- **`post_process()`**: Grid-search rounding thresholds on OOF to maximize QWK.

---

## 7. Key Tricks (ACTIONABLE)
- **If** image has black borders → **do** `crop_image_from_gray` before resize.
- **If** crop result is zero-dimension → **do** return original image unchanged.
- **If** applying Ben Graham → **do** use `sigmaX=10`; larger values (30/50) over-smooth.
- **If** using circle crop → **do** note peripheral lesions near disc margin may be clipped.
- **If** pretraining on 2015 DR data → **do** account for label inconsistency vs. Aravind graders.
- **If** using 2015 `.jpeg` images → **do** prefer `.png` equivalents to avoid compression artefacts.

---

## 8. Code Structure

```python
import os, cv2, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight, shuffle

SEED, IMG_SIZE, NUM_CLASSES = 77, 512, 5

def seed_everything(seed=SEED): pass
def load_data(train_csv, test_csv, img_dir): pass          # parse CSV → paths

def crop_image_from_gray(img, tol=7): pass                 # mask crop, zero-dim guard
def load_ben_color(path, sigmaX=10): pass                  # primary loader
def circle_crop(path, sigmaX=10): pass                     # optional variant

def create_folds(df, n_splits=5): pass                     # StratifiedKFold
def get_transforms(phase): pass                            # train/valid augmentations

class RetinopathyDataset: pass                             # load_ben_color → transform

def build_model(backbone='efficientnet-b4'): pass          # backbone + ordinal head
def criterion(preds, targets, class_weights=None): pass    # BCEWithLogits / MSE
def train_one_fold(fold, train_loader, val_loader, model, optimizer, scaler): pass
def validate(model, val_loader): pass                      # returns OOF preds + QWK

def predict(models, test_loader, tta=True): pass           # 5-fold + TTA ensemble
def post_process(oof_preds, oof_targets, test_preds): pass # threshold tuning on OOF

def main():
    seed_everything()
    df_train, df_test = load_data('train.csv', 'test.csv', 'train_images/')
    df_train = create_folds(df_train)
    models, oof_preds = [], []
    for fold in range(5):
        pass  # init loaders → build model → train_one_fold → append
    # test_preds = predict(models, test_loader)
    # final = post_process(oof_preds, oof_targets, test_preds)
    # pd.DataFrame({'id_code': df_test.id_code, 'diagnosis': final}).to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
```

---

## 9. Strategy Priority

1. **High Impact:** Ben Graham normalization · Gray-aware crop before resize · `IMG_SIZE=512`.
2. **Medium Impact:** Stratified 5-fold CV · Class-weighted loss · OOF threshold tuning for QWK.
3. **Minor:** Circle crop variant · `sigmaX=10` · Horizontal flip TTA · PNG over JPEG for 2015 data.