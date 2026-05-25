This document outlines the technical skills and methodologies implemented in the **HMS - Harmful Brain Activity Classification** competition, based on the provided KerasCV starter notebook.

---

## 🧠 Competition Objective
The goal is to classify seizures and other patterns of harmful brain activity in critically ill patients using EEG data and spectrograms.

## 🛠 Technical Skill Set

### 1. Deep Learning Frameworks & Backends
* **Keras 3 & KerasCV**: Leveraged for building computer vision pipelines with high-level abstractions.
* **Backend Agnosticism**: The implementation is designed to work across **JAX**, **TensorFlow**, and **PyTorch**, though JAX is the primary backend configured in this instance.
* **EfficientNetV2**: Utilized the `efficientnetv2_b2_imagenet` architecture as the core backbone for image-based classification.

### 2. Data Engineering & Processing
* **Format Conversion**: Expertly handled large-scale data by converting `.parquet` EEG spectrograms into `.npy` files to optimize I/O performance during training.
* **TF.Data Pipeline**: Constructed an efficient input pipeline using `tf.data.Dataset` for parallelized data loading and prefetching.
* **Signal Processing**: 
    * Converted raw signals into **Log Spectrograms**.
    * Applied normalization (mean/std) and padding to ensure consistent input shapes of $[400, 300]$.
    * Transformed mono-channel signals into 3-channel signals to maintain compatibility with pretrained **ImageNet** weights.

### 3. Advanced Model Training Techniques
* **Cross-Validation**: Implemented **StratifiedGroupKFold** (5 folds) to ensure that data from the same patient does not appear in both training and validation sets, preventing data leakage.
* **Augmentation Strategies**: Applied domain-specific augmentations including:
    * **MixUp**: For better generalization.
    * **RandomCutout**: Used for frequency and time masking to improve model robustness.
* **Learning Rate Scheduling**: Implemented a complex LR scheduler featuring a **Linear Warmup** followed by **Cosine Decay**.
* **Loss Function**: Directly optimized for the competition metric using **Kullback–Leibler (KL) Divergence** as the loss function.

### 4. Hardware & Optimization
* **GPU Acceleration**: Optimized code for execution on Kaggle GPU environments.
* **Parallelization**: Used `joblib` with the `loky` backend for high-speed multi-core preprocessing of spectrogram files.
* **Memory Management**: Configured data loading with `drop_remainder` and specific batch sizes (64) to manage VRAM efficiently.

---

## 📈 Evaluation & Results
* **Metric**: KL Divergence.
* **Artifacts**: The pipeline produces a `best_model.keras` file based on the lowest validation loss achieved during training.