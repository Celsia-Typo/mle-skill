## 🐾 Competition Objective
The goal is to identify animal species in camera trap images. This is a challenging computer vision task due to "empty" images (no animals), nighttime photography, and camouflage, requiring robust fine-grained visual classification.

## 🛠 Technical Skill Set

### 1. Computer Vision & Transfer Learning
* **Deep Convolutional Neural Networks (CNNs)**: Leveraged the **ResNet** architecture as a backbone for feature extraction.
* **Transfer Learning**: Implemented transfer learning by using weights pre-trained on **ImageNet**, allowing the model to leverage general visual features for a specific domain task.
* **Head Customization**: Modified the fully connected layers of the pre-trained model to match the specific number of animal categories in the iWildCam dataset.

### 2. PyTorch Framework Implementation
* **Custom Dataset Class**: Developed a flexible `Dataset` class using `torch.utils.data` to handle:
    * Loading images from paths using **PIL** and **OpenCV**.
    * Mapping complex JSON-based metadata (categories and image IDs) to training labels.
    * Handling truncated or corrupted images via `ImageFile.LOAD_TRUNCATED_IMAGES = True`.
* **DataLoader Optimization**: Configured data loaders with specific batch sizes and shuffling to optimize GPU memory utilization during training.
* **Tensor Manipulation**: Proficient in moving data between CPU and GPU (`.to(device)`) for accelerated computation.

### 3. Data Preprocessing & Augmentation
* **Dynamic Resizing**: Standardized input image dimensions (e.g., $224 \times 224$ or $256 \times 256$) to ensure compatibility with CNN input layers.
* **Normalization**: Applied ImageNet-standard mean and standard deviation normalization to align input data with the pre-trained model’s expected distribution.
* **Image Transformations**: Used `torchvision.transforms` for real-time data processing, ensuring consistent data flow from disk to the model.

### 4. Machine Learning Workflow
* **JSON Metadata Parsing**: Handled large-scale structured data by parsing competition-provided `.json` files to link animal species names to their numerical category IDs.
* **Validation Strategy**: Utilized `train_test_split` from `scikit-learn` to create a robust hold-out validation set for monitoring model performance and preventing overfitting.
* **Evaluation Metrics**: Focused on **F1-Score (Macro)**, the competition's primary metric, to account for the heavy class imbalance often found in camera trap data.

### 5. Efficient Pipeline & Deployment
* **Inference at Scale**: Built a prediction loop to process a large test set, converting raw model logits into final class predictions.
* **Submission Formatting**: Proficient in data post-processing using `pandas` to map numerical predictions back to original class strings and generate a competition-compliant `submission.csv`.
* **Logging**: Implemented custom logging functions (e.g., `kaggle_commit_logger`) to track training progress and system status within the Kaggle notebook environment.

---

## 📈 Key Outcomes
* **End-to-End Pipeline**: From raw JSON and image files to a final submission file.
* **Robustness**: Ability to handle imbalanced classes and noisy environmental data using modern deep learning techniques.