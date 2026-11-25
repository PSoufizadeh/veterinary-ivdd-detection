# Deep Learning for Disc Herniation Detection in Small Animals

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Deep%20Learning-blue)
![Research](https://img.shields.io/badge/Research-Published-success)
![Accuracy](https://img.shields.io/badge/Reported%20Test%20Accuracy-96%25-success)

## Overview

This repository contains the research implementation associated with the study **"Deep Learning for Diagnosis of Disc Herniation in Small Animals: A CNN-Based Approach Using CT Imaging"**, published in *Veterinary Radiology & Ultrasound* (2026).

The study investigates the application of deep learning to automated detection of intervertebral disc herniation in veterinary computed tomography (CT) images, using convolutional neural networks, transfer learning, and image segmentation.

The objective is to explore the potential of artificial intelligence as a diagnostic decision-support tool in veterinary medical imaging.

## Dataset

The original study included 1,651 CT images selected from 64,602 spinal CT images.

| Category | Number of Images |
|---|---:|
| Disc herniation | 818 |
| Healthy | 833 |
| Total | 1,651 |

The dataset was divided into:

- Training: 70%
- Validation: 15%
- Testing: 15%

The original dataset is confidential and is not included in this repository.

## Methodology

### Image Preprocessing

- Image resizing to 224 × 224 pixels
- Pixel intensity normalization
- Data augmentation
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- K-means clustering for synthetic mask generation

### Image Segmentation

A U-Net architecture was implemented for disc localization using synthetically generated segmentation masks.

### Image Classification

A VGG16-based convolutional neural network was fine-tuned using ImageNet-pretrained weights for binary classification of healthy and herniated discs.

The classification architecture incorporates:

- Transfer learning
- Global average pooling
- Batch normalization
- Fully connected layers
- Dropout and L2 regularization
- Sigmoid activation
- Adam optimizer

The segmentation and classification models are trained independently in the provided implementation.

## Published Results

The following results were reported in the published study.

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Disc herniation | 0.97 | 0.94 | 0.96 |
| Healthy | 0.95 | 0.98 | 0.96 |
| Macro average | 0.96 | 0.96 | 0.96 |
| Weighted average | 0.96 | 0.96 | 0.96 |

**Reported test accuracy: 96%**

**Peak validation accuracy: 95.12%**

## Repository Structure

    veterinary-ivdd-detection/
    ├── train.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore

## Installation

Install the required Python dependencies:

    pip install -r requirements.txt

## Usage

Place an authorized dataset in the following directory structure:

    veterinary-ivdd-detection/
    ├── Healthy/
    ├── IVDD/
    └── train.py

Run the training and evaluation pipeline:

    python train.py

The script performs dataset splitting, image preprocessing, synthetic mask generation, model training, and classification evaluation.

For subsequent runs, move the previously generated dataset directory before restarting the pipeline.

## Reproducibility and Limitations

- The original clinical dataset is not publicly available.
- The dataset originates from a single institution.
- The original implementation uses image-level splitting rather than patient-level splitting. Potential patient overlap between subsets cannot be excluded.
- External multicenter validation is required.
- The repository contains a technically refactored version of the research code.
- The published performance metrics have not been independently reproduced using the refactored implementation.
- The original training environment and exact dependency versions are not documented in this repository.

This implementation is intended for research and educational purposes and is not a clinically validated diagnostic system.

## Publication

Soufizadeh, P., Famil Ghadakchi, H., Modarres Tonekabony, S. H., & Molazem, M. (2026).

**Deep Learning for Diagnosis of Disc Herniation in Small Animals: A CNN-Based Approach Using CT Imaging.**

*Veterinary Radiology & Ultrasound*, 67, e70161.

DOI: [10.1111/vru.70161](https://doi.org/10.1111/vru.70161)

## Authors

- Parham Soufizadeh
- Hamed Famil Ghadakchi
- Seyyed Hossein Modarres Tonekabony
- Mohammad Molazem

## Research Areas

Artificial Intelligence · Deep Learning · Veterinary Medicine · Medical Imaging · Computer Vision · Digital Health · Clinical Decision Support

## Data Availability

The clinical imaging dataset is private and confidential. Access is restricted to the original research team in accordance with the ethical and data protection requirements of the study.
