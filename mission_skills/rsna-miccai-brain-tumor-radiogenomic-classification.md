# SCoT Prompt

---

## 1. Instructions

You are an expert deep learning engineer specializing in medical image analysis.
Your task is to implement components of a **brain tumor radiogenomic classification** pipeline
that predicts **MGMT promoter methylation status** (binary: 0 or 1) from multi-modal MRI brain
scans (FLAIR, T1w, T1wCE, T2w) using the RSNA-MICCAI dataset.

**Constraints and goals:**
- Input scans are stored as sequences of DICOM slices per patient per MRI type.
- Models must handle variable-length slice sequences via uniform temporal subsampling.
- Training must use stratified K-fold cross-validation (K=5) to prevent label leakage.
- Final predictions are produced by ensembling multiple fold models.
- Use PyTorch as the primary framework (EfficientNet backbone preferred).

**Before writing any code**, first sketch the problem-solving process as structured
pseudocode using **sequential**, **branch**, and **loop** structures. Then output the
final implementation as Python code with the pseudocode embedded as single-line comments.

---

## 2. Demonstration Examples

---

### Example 1 — DICOM Slice Loading and Preprocessing

```python
def load_dicom_volume(patient_dir: str, n_frames: int = 16, img_size: int = 256) -> torch.Tensor:
    """
    Load a sequence of DICOM slices from a patient directory,
    normalise pixel values to [0, 1], resize to img_size x img_size,
    and uniformly subsample to exactly n_frames slices.

    Args:
        patient_dir: Path to the directory containing .dcm slice files.
        n_frames:    Target number of frames after temporal subsampling.
        img_size:    Spatial resolution (H = W) for each frame.

    Returns:
        volume: FloatTensor of shape (n_frames, img_size, img_size).
    """
    # === SEQUENTIAL: collect and sort all DICOM file paths by slice index ===
    all_paths = sorted(
        glob.glob(os.path.join(patient_dir, "*.dcm")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1])
    )

    # === BRANCH: if directory is empty, return a zero volume ===
    if len(all_paths) == 0:
        return torch.zeros(n_frames, img_size, img_size)

    # === LOOP: read each DICOM, normalise, and resize to img_size ===
    frames = []
    for path in all_paths:
        dicom = pydicom.read_file(path)
        data = dicom.pixel_array.astype(np.float32)
        # SEQUENTIAL: shift min to 0, then scale max to 1
        data -= data.min()
        if data.max() != 0:
            data /= data.max()
        # SEQUENTIAL: resize spatial dims
        data = cv2.resize(data, (img_size, img_size))
        frames.append(torch.tensor(data))

    # === SEQUENTIAL: uniform temporal subsampling to exactly n_frames ===
    total = len(frames)
    # BRANCH: if fewer frames than needed, repeat the last frame to pad
    if total < n_frames:
        frames += [frames[-1]] * (n_frames - total)
        total = n_frames
    # SEQUENTIAL: pick n_frames indices evenly spaced across [0, total)
    indices = [int(i * total / n_frames) for i in range(n_frames)]
    volume = torch.stack([frames[i] for i in indices])   # (n_frames, H, W)

    return volume
```

---

### Example 2 — Multi-Modal MRI Dataset Class

```python
class BrainMRIDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that loads multi-modal MRI volumes for the RSNA-MICCAI task.

    For each patient the dataset stacks the four MRI types (FLAIR, T1w, T1wCE, T2w)
    along the channel axis, producing a 4-channel frame tensor.

    Args:
        df:          DataFrame with columns ['BraTS21ID', 'MGMT_value'].
        data_root:   Root directory containing train/ or test/ sub-folders.
        mri_types:   List of MRI type sub-folder names to load.
        n_frames:    Frames per volume after temporal subsampling.
        img_size:    Spatial size of each frame.
        transform:   Optional albumentations transform applied per-frame.
        is_train:    Whether to load from the train/ split (else test/).
    """

    def __init__(self, df, data_root, mri_types, n_frames=16,
                 img_size=256, transform=None, is_train=True):
        # SEQUENTIAL: store config
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.mri_types = mri_types
        self.n_frames = n_frames
        self.img_size = img_size
        self.transform = transform
        self.split = "train" if is_train else "test"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # SEQUENTIAL: retrieve patient id and label
        row = self.df.iloc[idx]
        patient_id = str(row["BraTS21ID"]).zfill(5)
        label = float(row.get("MGMT_value", -1))

        # LOOP: load one volume per MRI type and collect as channels
        channels = []
        for mri_type in self.mri_types:
            patient_dir = os.path.join(
                self.data_root, self.split, patient_id, mri_type
            )
            # SEQUENTIAL: load normalised volume → (n_frames, H, W)
            vol = load_dicom_volume(patient_dir, self.n_frames, self.img_size)
            channels.append(vol)

        # SEQUENTIAL: stack MRI types → (n_frames, n_channels, H, W)
        x = torch.stack(channels, dim=1)   # (T, C, H, W)

        # BRANCH: apply spatial augmentation per frame when transform is set
        if self.transform is not None:
            seed = random.randint(0, 99999)
            augmented = []
            for t in range(x.shape[0]):
                frame = x[t].permute(1, 2, 0).numpy()   # H×W×C
                random.seed(seed)
                frame = self.transform(image=frame)["image"]
                augmented.append(torch.tensor(frame).permute(2, 0, 1))
            x = torch.stack(augmented)   # (T, C, H, W)

        return {"X": x, "y": torch.tensor(label, dtype=torch.float32),
                "id": row["BraTS21ID"]}
```

---

### Example 3 — CNN + LSTM Model Definition and Single Epoch Training

```python
class CNN(nn.Module):
    """EfficientNet-B0 backbone that maps a (C, H, W) frame to a feature vector."""

    def __init__(self, n_channels: int = 4, cnn_features: int = 256):
        super().__init__()
        # SEQUENTIAL: project n_channels → 3 so EfficientNet accepts the input
        self.channel_map = nn.Conv2d(n_channels, 3, kernel_size=1)
        self.backbone = efficientnet_pytorch.EfficientNet.from_pretrained("efficientnet-b0")
        in_features = self.backbone._fc.in_features
        # SEQUENTIAL: replace classifier head with a feature projection layer
        self.backbone._fc = nn.Linear(in_features, cnn_features)

    def forward(self, x):
        # SEQUENTIAL: map channels, then extract spatial features
        x = F.relu(self.channel_map(x))       # (B, 3, H, W)
        return self.backbone(x)                # (B, cnn_features)


class CNNLSTMModel(nn.Module):
    """
    Sequence model: CNN extracts per-frame features, LSTM models temporal context,
    a linear head produces the binary logit.

    Args:
        n_channels:   Input MRI channels (one per MRI type).
        cnn_features: CNN output dimensionality.
        lstm_hidden:  LSTM hidden state size.
    """

    def __init__(self, n_channels=4, cnn_features=256, lstm_hidden=32):
        super().__init__()
        self.cnn = CNN(n_channels, cnn_features)
        self.lstm = nn.LSTM(cnn_features, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # LOOP: apply CNN to every frame independently
        frame_features = []
        for t in range(T):
            feat = self.cnn(x[:, t, :, :, :])   # (B, cnn_features)
            frame_features.append(feat)
        # SEQUENTIAL: stack frames → temporal sequence for LSTM
        seq = torch.stack(frame_features, dim=1)     # (B, T, cnn_features)
        _, (h_n, _) = self.lstm(seq)                 # h_n: (1, B, lstm_hidden)
        logit = self.head(h_n.squeeze(0))            # (B, 1)
        return logit


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Run one full training epoch.

    Args:
        model:     CNNLSTMModel in train mode.
        loader:    DataLoader yielding batches with keys 'X' and 'y'.
        optimizer: Torch optimiser.
        criterion: Loss function (BCEWithLogitsLoss).
        device:    Compute device.

    Returns:
        avg_loss: Mean loss over all batches.
    """
    model.train()
    total_loss = 0.0

    # LOOP: iterate over all mini-batches
    for batch in loader:
        X = batch["X"].to(device)    # (B, T, C, H, W)
        y = batch["y"].to(device)    # (B,)

        # SEQUENTIAL: forward pass → compute loss → backprop → update
        optimizer.zero_grad()
        logits = model(X).squeeze(1)   # (B,)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # SEQUENTIAL: return the mean batch loss
    avg_loss = total_loss / len(loader)
    return avg_loss
```

---

## 3. Testing Requirement（测试要求）

Now implement the **complete 5-fold stratified cross-validation training and ensemble inference pipeline** for the RSNA-MICCAI Brain Tumor Radiogenomic Classification task.

The pipeline must include:
- Stratified K-Fold (K=5) splitting on `MGMT_value`.
- Per-fold: instantiate `CNNLSTMModel`, train for `N_EPOCHS`, save the best checkpoint (by validation AUC).
- After all folds: load each saved checkpoint and run ensemble inference on the test set by averaging sigmoid outputs across all fold models.
- Write the final `submission.csv` with columns `BraTS21ID` and `MGMT_value`.

Let's think step by step.

Write your code here.
```python
# === Your implementation below ===
```
