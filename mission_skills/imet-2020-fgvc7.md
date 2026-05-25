Blueprint: Single-File .py Kaggle Solution Generator for iMet 2020 (PyTorch Multi-Label Inference)

Objective

Generate a clean, production-style single Python script (solution.py) for the Kaggle competition:

* Competition: iMet 2020 - FGVC7
* Task: Multi-label image classification
* Framework: PyTorch
* Backbone: ResNet50
* Goal: Reproduce inference pipeline similar to the provided notebook-based medal solution

The generated script must be fully self-contained and runnable as a single .py file inside Kaggle Notebook or Kaggle Script environment.

⸻

Global Requirements

File Constraints

Generate:

solution.py

Only one Python file.

No notebook cells.
No markdown.
No external modules beyond Kaggle default environment.

⸻

Core Functional Requirements

The generated script must include:

1. Imports
2. Configuration section
3. Logger utilities
4. Reproducibility utilities
5. Dataset classes
6. Albumentations transforms
7. Model architecture definitions
8. Model loading
9. Test inference loop
10. Thresholding
11. Submission generation
12. Main execution guard

⸻

Expected Pipeline

The pipeline should follow:

CSV submission template
    ↓
Test Dataset
    ↓
Albumentations preprocessing
    ↓
PyTorch DataLoader
    ↓
ResNet50 model
    ↓
Load trained checkpoint
    ↓
GPU inference
    ↓
Sigmoid activation
    ↓
Thresholding
    ↓
attribute_ids generation
    ↓
submission.csv

⸻

Detailed Architecture Specification

1. Imports Section

The script should import:

Standard libraries

os
gc
time
random
logging
pathlib.Path
contextlib.contextmanager
typing.Dict
functools.partial

Scientific stack

numpy
pandas
cv2
scipy
PIL.Image

PyTorch

torch
torch.nn as nn
torch.nn.functional as F
torchvision.models as models

Data utilities

Dataset
DataLoader

Augmentation

albumentations
albumentations.pytorch.ToTensorV2

Progress bar

tqdm

⸻

2. Configuration Section

Create centralized constants.

Required constants:

SEED = 777
N_CLASSES = 3474
HEIGHT = 128
WIDTH = 128
BATCH_SIZE = 128
THRESHOLD = 0.10
MODEL_PATH = "../input/imet2020/best-model.pt"
TEST_DIR = "../input/imet-2020-fgvc7/test"
SUBMISSION_PATH = "../input/imet-2020-fgvc7/sample_submission.csv"

Also define:

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

⸻

3. Logger Utilities

Generate:

init_logger()

Requirements:

* console + file logger
* DEBUG level
* formatted timestamps

timer() context manager

Should log:

[start]
[done in X sec]

⸻

4. Reproducibility

Generate:

seed_torch(seed)

Must set:

random.seed
numpy seed
torch.manual_seed
torch.cuda.manual_seed
PYTHONHASHSEED
torch.backends.cudnn.deterministic = True

Call it globally.

⸻

5. Dataset Classes

TestDataset

Must:

* inherit from Dataset
* receive dataframe + transform
* load images using OpenCV
* convert BGR → RGB
* apply Albumentations transforms
* return tensor image

Implementation details:

file_path = f"{TEST_DIR}/{image_id}.png"

⸻

6. Augmentation / Transform Pipeline

Generate:

get_transforms(data="valid")

Validation pipeline should include:

RandomCrop(256, 256)
HorizontalFlip(p=0.5)
Normalize(mean=..., std=...)
ToTensorV2()

Use ImageNet normalization:

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

⸻

7. Model Definitions

The script must define modular reusable architectures.

⸻

AvgPool Module

Generate custom adaptive pooling layer:

class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])

⸻

create_net()

Purpose:

* optionally load pretrained backbone weights from local Kaggle input folder

Logic:

if pretrained:
    load custom weights
else:
    use torchvision pretrained backbone

⸻

ResNet Wrapper

Requirements:

* support arbitrary ResNet backbone
* replace avgpool
* replace classifier head
* optional dropout head

Must support:

resnet18
resnet34
resnet50
resnet101
resnet152

Use:

functools.partial

⸻

DenseNet Wrapper

Also generate DenseNet support:

densenet121
densenet169
densenet201
densenet161

Even if unused.

⸻

8. Loss Function

Generate:

criterion = nn.BCEWithLogitsLoss(reduction="none")

Even if inference-only.

This mirrors training compatibility.

⸻

9. Model Initialization

Instantiate:

model = resnet50(num_classes=N_CLASSES, pretrained=True)

⸻

10. Checkpoint Loader

Generate:

load_model(model, path)

Requirements:

* load checkpoint
* load state['model']
* print epoch + step metadata

Expected checkpoint format:

{
    "model": ...,
    "epoch": ...,
    "step": ...
}

⸻

11. DataLoader Construction

Generate:

submission = pd.read_csv(SUBMISSION_PATH)
test_dataset = TestDataset(
    submission,
    transform=get_transforms(data="valid")
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

⸻

12. Inference Loop

Inference must:

* move model to GPU
* use torch.no_grad()
* sigmoid logits
* collect predictions in list
* concatenate NumPy arrays

Pseudo-flow:

preds = []
for images in test_loader:
    images = images.to(DEVICE)
    with torch.no_grad():
        logits = model(images)
    probs = torch.sigmoid(logits)
    preds.append(probs.cpu().numpy())

Use tqdm progress bar.

⸻

13. Prediction Postprocessing

After inference:

predictions = np.concatenate(preds) > THRESHOLD

For each row:

ids = np.nonzero(row)[0]

Convert to space-separated string:

" ".join(str(x) for x in ids)

Store into:

submission["attribute_ids"]

⸻

14. Submission Generation

Generate:

submission.to_csv("submission.csv", index=False)

Also print head of dataframe.

⸻

15. Main Function Structure

The generated script must use:

def main():
    ...
if __name__ == "__main__":
    main()

⸻

Code Quality Requirements

The generated .py must:

* be modular
* use type hints where reasonable
* avoid notebook-specific syntax
* avoid duplicated code
* be readable and competition-grade
* follow Kaggle style conventions

⸻

GPU / Performance Requirements

The generated solution should:

* support CUDA automatically
* use batch inference
* use pin_memory
* avoid gradient computation
* minimize CPU↔GPU transfers

⸻

Expected Output File

The final generated code must create:

submission.csv

compatible with Kaggle competition submission format.

⸻

Important Behavioral Constraints for the LLM

DO NOT

* generate notebook cells
* generate markdown explanations
* generate training loop
* generate validation metrics
* generate unsupported dependencies
* use Lightning / Hydra / timm
* split into multiple files

⸻

MUST INCLUDE

* exact inference logic
* exact thresholding behavior
* exact label formatting
* Albumentations preprocessing
* ResNet wrapper classes
* checkpoint loading logic
* tqdm progress bar
* reproducibility utilities

⸻

Desired Final Script Characteristics

The final generated solution.py should resemble:

* a strong Kaggle inference baseline
* compact but readable competition code
* reusable CV inference template
* production-style PyTorch inference pipeline