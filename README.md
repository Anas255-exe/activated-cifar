
# A Technical Report on Vision Transformer Implementation and Text-Driven Semantic Segmentation

## Abstract

This repository presents two projects in computer vision. The first project details the implementation and training of a Vision Transformer (ViT) architecture from scratch for the task of image classification on the CIFAR-10 dataset. The second project demonstrates a modular pipeline for text-driven semantic segmentation in both static images and video sequences, leveraging the capabilities of large-scale, pre-trained foundation models such as GroundingDINO and the Segment Anything Model 2 (SAM 2).

---

## Repository Contents

| File          | Description                                                                                                               |
| ------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **q1.ipynb**  | A PyTorch implementation of the Vision Transformer architecture, including the training and evaluation loop for CIFAR-10. |
| **q2.ipynb**  | An implementation of the text-driven segmentation pipeline for both single images and video sequences.                    |
| **README.md** | This document, providing a technical overview, execution protocol, and analysis for both projects.                        |

---

## Project 1: Vision Transformer for Image Classification

### Overview

This section outlines the from-scratch implementation of a Vision Transformer, designed for image classification on the CIFAR-10 benchmark. The model adheres to the architecture proposed by Dosovitskiy et al. in *“An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”* (ICLR 2021).

### Execution Protocol

To replicate the training and evaluation process, follow the steps below:

1. Open the `q1.ipynb` notebook in a Google Colab environment.
2. Ensure a GPU-accelerated runtime is selected (`Runtime > Change runtime type > GPU > T4`).
3. Execute all cells sequentially. All package dependencies, data loading, model training, and final evaluation are handled within the notebook.

---

### Final Model Configuration

```python
CONFIG = {
    'patch_size': 4,
    'embed_dim': 192,
    'depth': 6,
    'num_heads': 6,
    'mlp_ratio': 4,
    'drop_rate': 0.3,
    'attn_drop_rate': 0.3,
    'drop_path_rate': 0.2,
    'epochs': 40,
    'lr': 3e-4,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
}
```

The above configuration was selected to balance model capacity with the regularization required for the CIFAR-10 dataset. A compact ViT-Tiny variant was chosen to ensure stable training and to mitigate overfitting.

---

### Performance Evaluation

The model was trained for 40 epochs, and performance was measured by classification accuracy on the CIFAR-10 test set.

| Model Architecture                | Test Accuracy |
| --------------------------------- | ------------- |
| Vision Transformer (Custom, Tiny) | 82.67%        |

<img width="1677" height="562" alt="image" src="https://github.com/user-attachments/assets/ea3eebdb-f6b1-44f0-9612-e221ca89a0fc" />


Training and validation curves demonstrate stable convergence and effective learning over the training period.


---

### Analysis of Architectural and Training Considerations

**Architecture Sizing:**
A compact Vision Transformer with an embedding dimension of 192 and depth of 6 was implemented to align model capacity with dataset complexity. Larger models were avoided to reduce overfitting risks.

**Regularization Strategy:**
The model employed dropout and stochastic depth (DropPath) regularization to improve generalization. DropPath randomly deactivates residual blocks during training, enhancing robustness.

**Optimizer and Learning Rate Schedule:**
The AdamW optimizer was used with a linear warmup for the first five epochs, followed by a cosine annealing schedule. This combination ensures stable convergence, particularly during early training.

**Data Augmentation:**
Standard augmentation techniques, including random cropping and horizontal flipping, were applied to the training set to improve model robustness and generalization performance.

---

## Project 2: Text-Driven Semantic Segmentation Using Foundation Models

### Overview

This project demonstrates a pipeline for high-fidelity object segmentation from both images and videos, guided by natural language prompts. The system integrates two foundation models: GroundingDINO for zero-shot object detection and SAM 2 (Segment Anything Model 2) for class-agnostic segmentation.

---

### Part 1: Methodology for Single Image Segmentation

The single-image pipeline translates a textual description into a pixel-accurate segmentation mask through a multi-stage process:

1. **Input:** The system accepts an image and a corresponding text prompt describing the target object.
2. **Zero-Shot Detection:** The text prompt is processed by GroundingDINO to generate bounding box coordinates for the described object.
3. **Prompted Segmentation:** The bounding box serves as a spatial prompt for SAM 2, directing it toward the relevant region of interest.
4. **Mask Generation:** SAM 2 produces a high-resolution segmentation mask, which is overlaid on the original image for visualization.

*Figure 1: Schematic of the single-image segmentation pipeline, showing text input, bounding box detection, and mask generation.*

---

### Part 2: Extension to Video Object Segmentation

The single-image methodology was extended to enable continuous object segmentation across video frames.

1. **Initialization (First Frame):**
   The text prompt is used only on the initial frame to generate a reference mask using the GroundingDINO and SAM 2 pipeline.

2. **Mask Propagation (Subsequent Frames):**
   The segmentation mask from the previous frame is passed as a prompt to SAM 2 for the next frame. This exploits SAM 2’s ability to track and propagate segmentation masks efficiently.

3. **Video Compilation:**
   The sequence of frames with generated masks is compiled into a final video demonstrating continuous segmentation across time.

*Figure 2: Workflow of the video segmentation pipeline, illustrating text-based initialization and iterative mask propagation.*

---

### Limitations and Future Work

| Limitation                 | Description                                                                            | Future Direction                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Detector Dependency**    | The segmentation performance relies on accurate initial detections from GroundingDINO. | Incorporate periodic re-detection or ensemble object detectors to mitigate errors. |
| **Temporal Drift**         | Mask propagation can accumulate errors during object motion or occlusion.              | Integrate optical flow or periodic mask realignment strategies.                    |
| **Computational Overhead** | Large-scale models such as SAM 2 require high memory and GPU resources.                | Explore model distillation or quantization to reduce computational cost.           |

---

## Dependencies

The required dependencies are automatically installed in the provided notebooks. The primary libraries include:

```bash
torch
torchvision
opencv-python
matplotlib
timm
transformers
git+https://github.com/facebookresearch/segment-anything.git
```

---

## Summary

| Project                  | Objective                        | Key Components       | Outcome                                        |
| ------------------------ | -------------------------------- | -------------------- | ---------------------------------------------- |
| Vision Transformer (ViT) | Image classification on CIFAR-10 | Custom ViT, PyTorch  | Achieved stable accuracy and generalization    |
| Text-Driven Segmentation | Text-guided object segmentation  | GroundingDINO, SAM 2 | High-quality segmentation in images and videos |

---

## Author

**Mohammad Anas**
B.Tech in Computer Science and Engineering
Delhi Technological University (DTU)

---

## References

1. Dosovitskiy et al., *“An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,”* ICLR 2021.
2. IDEA Research, *GroundingDINO: Text-to-Box Detection Framework.*
3. Meta AI, *Segment Anything Model (SAM) and SAM 2.*

