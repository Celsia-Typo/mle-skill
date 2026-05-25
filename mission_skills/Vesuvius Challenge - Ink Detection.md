## 📜 Vesuvius Challenge - Ink Detection
**Objective:** Detect ink on carbonized papyrus fragments from 3D X-ray CT scans.

### Technical Skill Set
* **3D Volumetric Data Processing**:
    * Worked with high-resolution CT "slices" to identify sub-surface patterns.
    * Implemented **Z-slice selection** techniques to extract relevant layers from 3D volumes for 2D analysis.
* **Image Segmentation**:
    * Utilized a **U-Net** architecture (with a ResNet-18 backbone) for pixel-level classification of ink vs. no-ink.
* **Binary Masking & Spatial Analysis**: 
    * Processed large-scale binary masks to define training regions.
    * Applied **sliding window (tile-based)** training and inference to manage massive image dimensions that exceed GPU memory.
* **PyTorch Lightning**: Optimized the training workflow using **Lightning** for better code reproducibility and hardware scaling.
* **Submission Engineering**: Developed efficient RLE (Run-Length Encoding) scripts to compress high-resolution segmentation masks for competition submission.