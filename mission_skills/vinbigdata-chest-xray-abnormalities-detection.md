# VinBigData Chest X-ray Abnormalities Detection — Hint

> **Instructions for the LLM agent**: Your goal is to build a complete, end-to-end machine learning pipeline for thoracic abnormality detection from DICOM chest X-rays. The task combines a binary normality classifier (EfficientNet-B4) with a multi-class object detector (YOLOv11x), coupled with Weighted Boxes Fusion (WBF) annotation consolidation and ensemble. Before writing any code, first sketch the problem-solving logic using the three programming structures (sequential, branching, and looping structures) as inline comments inside each function. Then output the final implementation. Follow the SCoT comment format shown throughout this document exactly.

---

## §0 Blueprint

### §0.1 Task & Metric

- **Task**: Localize and classify 14 thoracic abnormalities in chest X-rays plus a "No Finding" (normal) class
- **Metric**: mAP over IoU [0.4, 0.45, 0.50, …, 0.75]; higher is better
- **Classes (14 abnormalities + 1 normal)**:
  `0: Aortic enlargement, 1: Atelectasis, 2: Calcification, 3: Cardiomegaly, 4: Consolidation, 5: ILD, 6: Infiltration, 7: Lung Opacity, 8: Nodule/Mass, 9: Other lesion, 10: Pleural effusion, 11: Pleural thickening, 12: Pneumothorax, 13: Pulmonary fibrosis, 14: No finding`
- **Input**: DICOM files + `train.csv` (multi-radiologist annotations) + `image_metadata_full.csv` (original width/height)
- **Output**: `submission.csv` with columns `image_id`, `PredictionString` — format: `"{class_id} {conf} {x_min} {y_min} {x_max} {y_max} ..."`
- **Normal label string**: `"14 1.0 0 0 1 1"` — use this whenever an image is judged as having no finding

### §0.2 Data Layout

```
./input/
  train.csv                    # image_id, class_name, class_id (0-14), rad_id,
                               #   x_min, y_min, x_max, y_max  (absolute pixels)
  sample_submission.csv        # image_id, PredictionString (placeholder)
  image_metadata_full.csv      # image_id, width, height  ← REQUIRED for rescaling
  train/   {image_id}.dicom    # chest X-ray DICOMs
  test/    {image_id}.dicom
```

### §0.3 Known Bugs & Critical Tricks

**Bug A — `MONOCHROME1` images must be inverted before use (FATAL)**
Some DICOM files have `PhotometricInterpretation == "MONOCHROME1"`, meaning pixel intensity is
inverted (bright = low signal). Failing to invert causes the model to see black lungs on a white background:
```python
img = apply_voi_lut(ds.pixel_array, ds)   # apply LUT first (Bug B)
if ds.PhotometricInterpretation == "MONOCHROME1":
    img = np.amax(img) - img              # ← invert (Bug A fix)
```

**Bug B — VOI LUT must be applied BEFORE any normalization (FATAL)**
`apply_voi_lut` maps raw pixel values to the display-intended range. Normalizing first and applying LUT
second produces garbage outputs:
```python
# WRONG:
img = (ds.pixel_array / ds.pixel_array.max() * 255).astype(np.uint8)
# CORRECT:
img = apply_voi_lut(ds.pixel_array, ds)   # apply LUT on raw array first
# then invert if MONOCHROME1, then normalize to [0,255] uint8
```

**Bug C — Letterbox bounding box transform: pad must be ADDED after scale (FATAL)**
When letterboxing an image to `target_size`, box coordinates must be transformed consistently:
```python
# WRONG:
new_x_min = boxes[:, 0] + pad_x   # forgetting to scale first
# CORRECT:
scale  = target_size / max(orig_h, orig_w)
pad_x  = (target_size - orig_w * scale) / 2
pad_y  = (target_size - orig_h * scale) / 2
new_boxes[:, 0] = boxes[:, 0] * scale + pad_x   # scale THEN pad
new_boxes[:, 1] = boxes[:, 1] * scale + pad_y
new_boxes[:, 2] = boxes[:, 2] * scale + pad_x
new_boxes[:, 3] = boxes[:, 3] * scale + pad_y
```

**Bug D — YOLO training labels must skip class_id 14 (No Finding) (FATAL)**
YOLO detectors operate only on abnormality classes (0–13). Writing class 14 into the label file
causes the model to try to detect a "normal" region with a dummy bounding box:
```python
for box in boxes:
    cls_id = int(box[4])
    if cls_id == 14:
        continue    # ← skip No Finding; never write to YOLO label file
    f.write(f"{cls_id} {xc} {yc} {bw} {bh}\n")
```

**Bug E — WBF requires normalized [0, 1] coordinates; always denormalize after (FATAL)**
`weighted_boxes_fusion` operates on boxes in `[0, 1]` range. Passing absolute pixel coordinates
silently produces nonsense output (boxes clipped to [0,1]):
```python
# Normalize before WBF:
b[:, [0, 2]] /= width;  b[:, [1, 3]] /= height
merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(...)
# Denormalize after WBF:
merged_boxes[:, [0, 2]] *= width;  merged_boxes[:, [1, 3]] *= height
```

**Trick 1 — StratifiedGroupKFold: build composite string label from multi-label indicator**
`StratifiedGroupKFold` needs a 1-D `y_stratify`. Build a binary indicator matrix (N × 14) and
convert each row to a string to represent the unique combination of findings:
```python
indicators = np.zeros((n_samples, 14), dtype=np.int8)
for i, boxes in enumerate(y):
    classes = boxes[:, 4].astype(int)
    for c in classes:
        if 0 <= c < 14:
            indicators[i, c] = 1
y_stratify = np.array(["".join(map(str, row)) for row in indicators])
```

**Trick 2 — Binary classifier input: repeat grayscale to 3 channels, resize to 512×512**
EfficientNet-B4 expects 3-channel input. The preprocessed tensor is grayscale `(1, 1024, 1024)`.
Repeat the channel dimension and downsample for training efficiency:
```python
img = images[i].repeat(3, 1, 1)       # (1,H,W) → (3,H,W)
img = F.interpolate(img.unsqueeze(0), size=(512, 512), mode='bilinear').squeeze(0)
```

**Trick 3 — YOLO inference conf=0.001 (very low); let WBF ensemble filter**
Running YOLO with the default `conf=0.25` discards many true positives before ensemble.
Use `conf=0.001` to preserve all candidates and let downstream WBF do the filtering:
```python
results = yolo_model.predict(img_rgb, imgsz=1024, conf=0.001, verbose=False)
```

**Trick 4 — Binary classifier gate: prob_normal > 0.95 → output normal label immediately**
Before running the expensive YOLO detector, check if the classifier is confident the image is
normal. This saves compute and avoids spurious detections on clean images:
```python
prob_normal = torch.sigmoid(classifier(cls_img)).item()
if prob_normal > 0.95:
    preds.append("14 1.0 0 0 1 1")
    continue
# else: run YOLO
```

**Trick 5 — Letterbox inversion in workflow: invert pad AND scale to recover original coords**
After ensemble, boxes are in 1024×1024 letterbox space. Invert the letterbox transform using
the stored `scale`, `pad_x`, `pad_y` to recover original image coordinates:
```python
scale = 1024.0 / max(orig_h, orig_w)
pad_x = (1024 - int(orig_w * scale)) // 2
pad_y = (1024 - int(orig_h * scale)) // 2
x1_orig = (x1_1024 - pad_x) / scale
y1_orig = (y1_1024 - pad_y) / scale
```

**Trick 6 — Memory management between folds: gc.collect() + torch.cuda.empty_cache()**
Each fold creates large tensors. Without explicit cleanup, CUDA OOM errors occur by fold 3:
```python
del X_tr_p, y_tr_p, X_val_p, y_val_p
gc.collect()
torch.cuda.empty_cache()
```

**Trick 7 — Final submission: merge with sample_submission to guarantee order and coverage**
Do not just sort by `image_id`; left-join against `sample_submission.csv` to preserve the
exact row order and fill any missing predictions with the normal label:
```python
final_submission = sample_sub[['image_id']].merge(submission, on='image_id', how='left')
final_submission['PredictionString'] = final_submission['PredictionString'].fillna("14 1.0 0 0 1 1")
```

---

## §1 `load_data`

```python
# Input:  validation_mode: bool  (True → return only 200 samples for fast debug)
# Output: (X_train: pd.DataFrame,        # metadata + filepath column
#           y_train: List[np.ndarray],    # each array: [x_min,y_min,x_max,y_max,class_id]
#           X_test:  pd.DataFrame,
#           test_ids: pd.Series)

# Constraints & Tricks:
#   Bug A: MONOCHROME1 → invert before normalizing
#   Bug B: apply_voi_lut BEFORE any normalization
#   Bug E: WBF annotation consolidation requires normalized coords
#   Trick: 36 parallel workers (ProcessPoolExecutor) for DICOM→PNG conversion
#   Metadata: read width/height from image_metadata_full.csv (not from train.csv)

# Sequential:
#   step 1 → define paths: train_dicom_dir, test_dicom_dir, train_csv, meta_csv
#   step 2 → makedirs: train_png_dir, test_png_dir
#   step 3 → build processing_tasks = [(dicom_path, png_path), ...]
#             for both train and test DICOMs
#   step 4 → Branch: if processing_tasks exist →
#               ProcessPoolExecutor(max_workers=36).map(_process_single_dicom, tasks)
#   step 5 → Branch: if consensus_csv does not exist →
#               consolidate_annotations(train_csv, meta_csv) → consensus_df
#             else → pd.read_csv(consensus_csv_path)
#   step 6 → meta_df = pd.read_csv(meta_csv_path)
#   step 7 → X_train = meta_df filtered to available PNGs; add 'filepath' column
#   step 8 → y_train = [consensus_df rows for each image_id as np.ndarray]
#   step 9 → X_test = meta_df filtered to test PNGs; add 'filepath' column
#   step 10 → test_ids = X_test['image_id']
#   step 11 → Branch: if validation_mode → head(200) for all

# Branch:
#   if ds.PhotometricInterpretation == "MONOCHROME1" → img = np.amax(img) - img  (Bug A)
#   if np.max(img) > 0 → img = (img / np.max(img) * 255).astype(np.uint8)
#   else               → img = np.zeros_like(img, dtype=np.uint8)
#   if consensus_csv exists → skip re-consolidation

# Helper: _process_single_dicom(dicom_path, png_path)  [runs in worker process]
# Sequential:
#   step 1 → if png_path exists: return  (skip already-converted)
#   step 2 → ds = pydicom.dcmread(dicom_path)
#   step 3 → img = apply_voi_lut(ds.pixel_array, ds)   # Bug B: LUT first
#   step 4 → Branch: if MONOCHROME1 → img = np.amax(img) - img   # Bug A
#   step 5 → img = img.astype(float32); img -= img.min()
#   step 6 → Branch: if img.max() > 0 → img = (img / img.max() * 255).astype(uint8)
#   step 7 → cv2.imwrite(png_path, img)

# Helper: consolidate_annotations(train_csv, meta_csv) → consensus_df
# Sequential:
#   step 1 → train_df = pd.read_csv(train_csv); merge width/height from meta_csv
#   step 2 → for each unique image_id:
#               findings = rows where class_id != 14
#               Branch: if len(findings) == 0 → append normal row {class_id:14, box:[0,0,1,1]}
#               else:
#                 for each rad_id in findings.rad_id.unique():
#                   normalize boxes to [0,1]  (Bug E)
#                   append to boxes_list, scores_list=[1.0]*N, labels_list
#                 merged = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
#                                               iou_thr=0.5, skip_box_thr=0.0001)   # Bug E
#                 denormalize merged boxes back to pixels
#   step 3 → consensus_df = pd.DataFrame(results); save to consensus_csv_path
```

---

## §2 `preprocess`

```python
# Input:  X_train: pd.DataFrame, y_train: List[np.ndarray],
#         X_val:   pd.DataFrame, y_val:   List[np.ndarray],
#         X_test:  pd.DataFrame
# Output: (X_train_p: torch.Tensor (N,1,1024,1024),  y_train_p: List[np.ndarray],
#           X_val_p,   y_val_p,   X_test_p)

# Constraints & Tricks:
#   target_size = 1024
#   num_workers = 36  (ProcessPoolExecutor)
#   CLAHE: clipLimit=2.0, tileGridSize=(8,8)
#   Letterbox: maintain aspect ratio, pad remainder with zeros
#   Bug C: new_box = old_box * scale + pad   (scale BEFORE adding pad)
#   Normalize tensor to [0,1] float32; add channel dim → (N,1,H,W)

# Sequential:
#   step 1 → for split in [train, val, test]:
#               tasks = [(filepath, orig_w, orig_h, 1024), ...]
#               ProcessPoolExecutor(36).map(_preprocess_single_image, tasks)
#               stack → np.ndarray (N, 1024, 1024)
#               tensor = torch.from_numpy(...).unsqueeze(1).float() / 255.0  → (N,1,1024,1024)
#               assert not tensor.isnan().any() and not tensor.isinf().any()
#   step 2 → y_train_p = _transform_labels(y_train, X_train, target_size=1024)
#   step 3 → y_val_p   = _transform_labels(y_val,   X_val,   target_size=1024)
#   step 4 → assert len(X_train_p) == len(y_train_p)
#             assert len(X_val_p)   == len(y_val_p)

# Helper: _preprocess_single_image(filepath, orig_w, orig_h, target_size) → np.ndarray
# Sequential:
#   step 1 → img = cv2.imread(filepath, IMREAD_GRAYSCALE)
#   step 2 → clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); img = clahe.apply(img)
#   step 3 → scale = target_size / max(orig_h, orig_w)
#   step 4 → new_w, new_h = int(orig_w*scale), int(orig_h*scale)
#   step 5 → img_resized = cv2.resize(img, (new_w, new_h), interpolation=INTER_LINEAR)
#   step 6 → canvas = np.zeros((target_size, target_size), uint8)
#   step 7 → pad_x = (target_size - new_w) // 2; pad_y = (target_size - new_h) // 2
#   step 8 → canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized
#   step 9 → return canvas

# Helper: _transform_labels(y, X_meta, target_size) → List[np.ndarray]   (Bug C fix)
# Loop: for i, boxes in enumerate(y):
#   step 1 → orig_w = X_meta.iloc[i]['width']; orig_h = X_meta.iloc[i]['height']
#   step 2 → scale = target_size / max(orig_h, orig_w)
#   step 3 → pad_x = (target_size - orig_w*scale) / 2
#   step 4 → pad_y = (target_size - orig_h*scale) / 2
#   step 5 → new_boxes = boxes.copy().astype(float32)
#   step 6 → new_boxes[:,0] = boxes[:,0]*scale + pad_x   # x_min (scale then pad — Bug C)
#   step 7 → new_boxes[:,1] = boxes[:,1]*scale + pad_y   # y_min
#   step 8 → new_boxes[:,2] = boxes[:,2]*scale + pad_x   # x_max
#   step 9 → new_boxes[:,3] = boxes[:,3]*scale + pad_y   # y_max
#   step 10→ new_boxes[:,:4] = np.clip(new_boxes[:,:4], 0, target_size)
```

---

## §3 `get_splitter`

```python
# Input:  X: pd.DataFrame  (must contain 'image_id' column)
#         y: List[np.ndarray]  (boxes with class_id in column 4)
# Output: VinBigDataSplitter (sklearn-compatible, .split(X, y) → Iterator[(train_idx, val_idx)])

# Constraints & Tricks:
#   Trick 1: StratifiedGroupKFold — composite string label to stratify on multi-label findings
#   Group by image_id to prevent same image appearing in both train and val splits
#   n_splits=5, shuffle=True, random_state=42

# Sequential (VinBigDataSplitter.split):
#   step 1 → num_classes=14; indicators = np.zeros((n_samples, 14), int8)
#   Loop:    for i, boxes in enumerate(y):
#     step 2 →   classes = boxes[:,4].astype(int)
#     step 3 →   for c in classes: if 0 <= c < 14: indicators[i,c] = 1
#   step 4 → y_stratify = ["".join(map(str, row)) for row in indicators]   # Trick 1
#   step 5 → groups = X['image_id'].values
#   step 6 → return StratifiedGroupKFold(5, shuffle=True, random_state=42)
#                      .split(X, y_stratify, groups=groups)

# Branch:
#   if 'image_id' not in X.columns → raise KeyError
#   if groups explicitly passed   → use passed groups; otherwise default to X['image_id']
```

---

## §4 `train_and_predict`

### §4.1 `BinaryClassifierDataset`

```python
# Input:  images: torch.Tensor (N,1,1024,1024), labels: List[np.ndarray]
# Output: (img: Tensor (3,512,512), target: float32 scalar)  per item

# Constraints & Tricks:
#   Trick 2: grayscale (1,1024,1024) → repeat → (3,1024,1024) → resize → (3,512,512)
#   target = 1.0 if any box has class_id == 14 (No Finding), else 0.0

# Sequential (__init__):
#   step 1 → self.targets = [1 if 14 in boxes[:,4] else 0 for boxes in labels]
#   step 2 → self.targets = torch.tensor(self.targets, dtype=float32)

# Sequential (__getitem__):
#   step 1 → img = self.images[idx]                    # (1, 1024, 1024)
#   step 2 → img = img.repeat(3, 1, 1)                 # (3, 1024, 1024)  Trick 2
#   step 3 → img = F.interpolate(img.unsqueeze(0), (512,512), mode='bilinear').squeeze(0)
#   step 4 → return img, self.targets[idx]
```

### §4.2 Classifier Training (with DDP)

```python
# Input:  X_train: Tensor, y_train: List[np.ndarray],
#         classifier_weights: str  (save path)
# Output: saved EfficientNet-B4 state_dict

# Constraints & Tricks:
#   DDP: mp.spawn → train_classifier_worker per GPU
#   Single-GPU fallback: direct training loop
#   optimizer: AdamW(lr=1e-4); criterion: BCEWithLogitsLoss
#   epochs = 5; batch_size = 8

# Sequential (train_classifier_worker, rank, world_size, ...):
#   step 1 → setup_ddp(rank, world_size)
#   step 2 → dataset = BinaryClassifierDataset(images, labels)
#   step 3 → sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
#   step 4 → loader  = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=4)
#   step 5 → model   = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1).to(rank)
#   step 6 → model   = DDP(model, device_ids=[rank])
#   Loop: for epoch in range(5):
#     step 7 →   sampler.set_epoch(epoch)
#     Loop:      for imgs, targets in loader:
#       step 8 →     loss = criterion(model(imgs.to(rank)), targets.to(rank).unsqueeze(1))
#       step 9 →     loss.backward(); optimizer.step(); optimizer.zero_grad()
#   step 10 → if rank == 0: torch.save(model.module.state_dict(), classifier_weights)
#   step 11 → cleanup_ddp()

# Branch:
#   if num_gpus > 1 → mp.spawn(train_classifier_worker, nprocs=num_gpus)
#   else            → single-GPU loop (same logic, no DDP)
```

### §4.3 YOLOv11x Detector Training

```python
# Input:  X_train: Tensor, y_train: List[np.ndarray], tmp_dir: str
# Output: trained YOLOv11x weights in tmp_dir/yolo_vinbigdata/

# Constraints & Tricks:
#   Bug D: skip class_id 14 when writing YOLO label files
#   YOLO format: [class_id  x_center  y_center  width  height]  normalized to [0,1]
#   imgsz=1024, batch=8, epochs=50, SGD lr0=0.01, cos_lr=True
#   mosaic=1.0, mixup=0.8  (strong augmentation for medical images)

# Sequential:
#   step 1 → makedirs: yolo_data_dir/images/train, yolo_data_dir/labels/train
#   Loop:    for i in range(len(X_train)):
#     step 2 →   img_np = (X_train[i,0].numpy()*255).astype(uint8)
#     step 3 →   cv2.imwrite(image_path, img_np)
#     step 4 →   open label_path for writing
#     Loop:      for box in y_train[i]:
#       step 5 →     cls_id = int(box[4])
#       step 6 →     Branch: if cls_id == 14: continue   # Bug D: skip No Finding
#       step 7 →     xc = (box[0]+box[2])/2/1024.0; yc = (box[1]+box[3])/2/1024.0
#       step 8 →     bw = (box[2]-box[0])/1024.0;   bh = (box[3]-box[1])/1024.0
#       step 9 →     f.write(f"{cls_id} {xc} {yc} {bw} {bh}\n")
#   step 10 → write data.yaml (path, train, val, nc=14, names=[...])
#   step 11 → yolo = YOLO("yolo11x.pt")
#   step 12 → yolo.train(data=yaml_path, epochs=50, imgsz=1024, batch=8,
#                         device=list(range(num_gpus)), optimizer='SGD', lr0=0.01,
#                         cos_lr=True, mosaic=1.0, mixup=0.8,
#                         project=tmp_dir, name="yolo_vinbigdata")
```

### §4.4 Inference (classifier gate → YOLO detect)

```python
# Input:  images: torch.Tensor (N,1,1024,1024), classifier, yolo_model, device
# Output: List[str]  prediction strings per image

# Constraints & Tricks:
#   Trick 3: YOLO conf=0.001 (very low); WBF ensemble filters later
#   Trick 4: classifier gate prob_normal > 0.95 → skip YOLO, output normal label
#   grayscale → RGB via cv2.cvtColor(img_np, COLOR_GRAY2RGB) for YOLO input

# Loop: for i in range(len(images)):
#   step 1 → img_tensor = images[i].to(device).unsqueeze(0)
#   step 2 → cls_img = F.interpolate(img_tensor.repeat(1,3,1,1), (512,512))
#   step 3 → prob_normal = sigmoid(classifier(cls_img)).item()
#   step 4 → Branch: if prob_normal > 0.95 →   # Trick 4
#                 preds.append("14 1.0 0 0 1 1"); continue
#   step 5 → img_np  = (images[i,0].numpy()*255).astype(uint8)
#   step 6 → img_rgb = cv2.cvtColor(img_np, COLOR_GRAY2RGB)
#   step 7 → results = yolo_model.predict(img_rgb, imgsz=1024, conf=0.001, verbose=False)[0]  # Trick 3
#   step 8 → Branch: if len(results.boxes) == 0 → preds.append("14 1.0 0 0 1 1")
#             else:
#     Loop:    for box in results.boxes:
#       step 9 →   cls, conf, xyxy = int(cls[0]), float(conf[0]), xyxy[0].cpu().numpy()
#       step 10 →  res.append(f"{cls} {conf:.4f} {int(xyxy[0])} {int(xyxy[1])} {int(xyxy[2])} {int(xyxy[3])}")
#     step 11 →  preds.append(" ".join(res))
```

---

## §5 `ensemble`

```python
# Input:  all_val_preds:  Dict[str, List[str]]  (fold_name → predictions per image)
#         all_test_preds: Dict[str, List[str]]
#         y_val: List[np.ndarray]  (ground truth, for optional scoring)
# Output: List[str]  final test prediction strings

# Constraints & Tricks:
#   Bug E: normalize boxes to [0,1] before WBF; denormalize to 1024 after
#   IOU_THR=0.5, SKIP_BOX_THR=0.001, equal model weights
#   If all folds predict "No finding" → output "14 1.0 0 0 1 1"

# Sequential:
#   step 1 → model_names = list(all_test_preds.keys())
#   step 2 → WEIGHTS = [1.0] * len(model_names)

# Loop: for idx in range(num_samples):
#   step 3 → boxes_list=[], scores_list=[], labels_list=[]
#   Loop:    for model_name in model_names:
#     step 4 →   b, s, l = _parse_pred_string(all_test_preds[model_name][idx])
#     step 5 →   boxes_list.append(b); scores_list.append(s); labels_list.append(l)
#   step 6 → Branch: if not any(len(b)>0 for b in boxes_list) →
#                 final_test_preds.append("14 1.0 0 0 1 1"); continue
#   step 7 → fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
#               boxes_list, scores_list, labels_list,
#               weights=WEIGHTS, iou_thr=0.5, skip_box_thr=0.001)   # Bug E: already normalized
#   step 8 → Branch: if len(fused_boxes) == 0 → append "14 1.0 0 0 1 1"
#   step 9 → else:
#               box_1024 = np.clip(fused_boxes, 0, 1) * 1024.0
#               res = [f"{int(cls)} {conf:.4f} {round(x1)} {round(y1)} {round(x2)} {round(y2)}"]
#               final_test_preds.append(" ".join(res))

# Helper: _parse_pred_string(pred_str) → (boxes, scores, labels)
# Sequential:
#   step 1 → if pred_str is None or pred_str.strip() == "14 1.0 0 0 1 1" → return [], [], []
#   step 2 → parts = pred_str.split()
#   Loop:    for i in range(0, len(parts), 6):
#     step 3 →   cls=int(parts[i]); if cls==14: continue
#     step 4 →   conf=float(parts[i+1])
#     step 5 →   boxes.append([xmin/1024, ymin/1024, xmax/1024, ymax/1024])  # normalize
#     step 6 →   scores.append(conf); labels.append(cls)
```

---

## §6 `workflow`

```python
# Input:  (none — reads all from ./input/)
# Output: submission.csv at OUTPUT_DATA_PATH

# Constraints & Tricks:
#   Trick 5: invert letterbox to recover original image coordinates
#   Trick 6: gc.collect() + torch.cuda.empty_cache() after each fold
#   Trick 7: merge with sample_submission (not sort) for correct row order

# Sequential:
#   step 1  → torch.cuda.empty_cache(); gc.collect()
#   step 2  → X_train_full, y_train_full, X_test, test_ids = load_data(validation_mode=False)
#   step 3  → preprocess all at once (pass dummy val to satisfy signature):
#               X_train_p_all, y_train_p_all, _, _, X_test_p =
#                   preprocess(X_train_full, y_train_full,
#                              X_train_full.head(1), [y_train_full[0]], X_test)
#   step 4  → splitter = get_splitter(X_train_full, y_train_full)
#   step 5  → engine = PREDICTION_ENGINES["effnet_yolov11_2stage"]
#   step 6  → all_val_preds={}; all_test_preds={}; oof=[None]*len(X_train_full)

#   Loop:   for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
#     step 7  →   X_tr_p = X_train_p_all[train_idx]
#     step 8  →   y_tr_p = [y_train_p_all[i] for i in train_idx]
#     step 9  →   X_val_p = X_train_p_all[val_idx]
#     step 10 →   y_val_p = [y_train_p_all[i] for i in val_idx]
#     step 11 →   val_preds, test_preds = engine(X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p)
#     step 12 →   all_val_preds[f"fold_{fold}"] = val_preds
#     step 13 →   all_test_preds[f"fold_{fold}"] = test_preds
#     step 14 →   for i, idx in enumerate(val_idx): oof[idx] = val_preds[i]
#     step 15 →   del X_tr_p, y_tr_p, X_val_p, y_val_p   # Trick 6
#     step 16 →   gc.collect(); torch.cuda.empty_cache()

#   step 17 → final_preds_1024 = ensemble(all_val_preds, all_test_preds, y_train_full)

#   step 18 → rescale_predictions: invert letterbox for each test image (Trick 5)
#   Loop:     for i, pred_str in enumerate(final_preds_1024):
#     step 19 →   orig_w = X_test.iloc[i]['width']; orig_h = X_test.iloc[i]['height']
#     step 20 →   Branch: if pred_str == "14 1.0 0 0 1 1" → keep as is; continue
#     step 21 →   scale = 1024.0 / max(orig_h, orig_w)
#     step 22 →   pad_x = (1024 - int(orig_w*scale)) // 2
#     step 23 →   pad_y = (1024 - int(orig_h*scale)) // 2
#     step 24 →   x1_orig = (x1_1024 - pad_x) / scale  (invert letterbox)
#     step 25 →   clip to [0, orig_w] and [0, orig_h]

#   step 26 → submission = pd.DataFrame({'image_id': test_ids, 'PredictionString': rescaled})
#   step 27 → final_submission = sample_sub[['image_id']].merge(submission, on='image_id', how='left')  # Trick 7
#   step 28 → final_submission['PredictionString'].fillna("14 1.0 0 0 1 1", inplace=True)
#   step 29 → final_submission.to_csv(submission_path, index=False)
```

---

## §7 Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `target_size` | `1024` | Letterbox target; single resolution |
| `CLAHE clipLimit` | `2.0` | Adaptive histogram equalization |
| `CLAHE tileGridSize` | `(8, 8)` | |
| `NUM_WORKERS` | `36` | ProcessPoolExecutor for DICOM & preprocess |
| `WBF iou_thr` | `0.5` | Annotation consolidation and ensemble |
| `WBF skip_box_thr` | `0.001` | Minimum confidence to participate in WBF |
| `n_splits` | `5` | StratifiedGroupKFold |
| `classifier_epochs` | `5` | EfficientNet-B4 binary classifier |
| `classifier_lr` | `1e-4` | AdamW |
| `classifier_batch` | `8` | |
| `classifier_input` | `512×512` | Downsampled from 1024 for efficiency |
| `yolo_model` | `yolo11x.pt` | YOLOv11x (largest variant) |
| `yolo_epochs` | `50` | |
| `yolo_imgsz` | `1024` | |
| `yolo_batch` | `8` | |
| `yolo_lr0` | `0.01` | SGD with cosine LR decay |
| `yolo_mosaic` | `1.0` | |
| `yolo_mixup` | `0.8` | |
| `yolo_conf_infer` | `0.001` | Very low; WBF filters (Trick 3) |
| `normal_gate_thresh` | `0.95` | Classifier prob to suppress YOLO (Trick 4) |
| `SEED` | `42` | |

---

## §8 What Didn't Work — Do NOT Repeat These

**❌ Per-box filtering with the classifier**
```python
# WRONG: using the image-level classifier to filter individual boxes
for box in ensemble_boxes:
    if classifier_prob[box.class_id] < threshold:
        remove(box)   # severe missed detections; verified as ineffective
# CORRECT: classifier is used only as an image-level normality gate (Trick 4)
```

**❌ Cropped-box classifier (Bbox Filter)**
```python
# WRONG: train EfficientNet-B6 on cropped box regions to filter false positives
crop = image[y_min:y_max, x_min:x_max]
if efficientnet_b6(crop).pred != class_id:
    remove(box)   # local crops lack global context; unstable; verified as ineffective
# CORRECT: if secondary verification is needed, feed the full image + box coords
```

**❌ External dataset pretraining (NIH ChestX-ray)**
```python
# NOT RECOMMENDED: NIH class definitions differ significantly from VinBigData
backbone = pretrain_on_NIH(nih_dataset, classes=14)
detector.backbone.load_weights(backbone)
train_detection_heads_only(competition_data)   # mAP not significantly improved
# CORRECT: ImageNet pretrained + end-to-end fine-tuning on competition data directly
```

**❌ Class-wise detectors**
```python
# NOT RECOMMENDED: separate detector per class leads to inconsistent CV splits and no net gain
for class_id in range(14):
    dataset = filter_by_class(train_data, class_id)   # clsX + cls14
    model = train_detector(dataset, class_specific_anchors)
# CORRECT: single multi-class detector on unified dataset; keep CV splits consistent
```

---

## §9 Trigger Prompt

> Let's think step by step. For each function (`load_data`, `preprocess`, `get_splitter`, `train_and_predict`, `ensemble`, `workflow`), first write the SCoT structure as inline comments (Input / Output / Constraints & Tricks / Sequential / Branch / Loop), then write your code here.
