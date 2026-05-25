## 🔬 BMS - Molecular Translation
**Objective:** Transform images of chemical compounds into InChI (International Chemical Identifier) strings.

### Technical Skill Set
* **Computer Vision & OCR**: Developed a pipeline to interpret complex chemical structures from images.
* **Sequence-to-Sequence Modeling**: Implemented an Encoder-Decoder architecture using **ResNet** as the visual encoder and **LSTMs** for text/string generation.
* **Natural Language Processing (NLP)**:
    * Performed tokenization and vocabulary building for chemical notation (InChI strings).
    * Managed special tokens for sequence start, end, and padding.
* **PyTorch Deep Learning**:
    * Customized **Dataset** and **DataLoader** classes for high-volume image-text pairs.
    * Applied image augmentations using the **Albumentations** library to improve model generalization.
* **Performance Optimization**: Utilized a "Reduced" dataset approach for faster prototyping and iterative model testing.

