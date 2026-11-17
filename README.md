# Sentinel-2 LULC Classification Pipeline

An end-to-end Deep Learning pipeline built in pure PyTorch for Land Use and Land Cover (LULC) classification using optical Earth Observation data. 

This project processes 64x64 pixel patches (10m Ground Sample Distance) from the Sentinel-2 Multispectral Instrument (MSI) to classify 10 distinct land cover classes, leveraging the EuroSAT dataset.

## Architecture

The model (`EuroSatCNN`) features a custom Convolutional Neural Network architecture designed specifically for satellite imagery analysis:
- **Feature Extraction:** 3 modular `CNNBlock` stages. Each block encapsulates a `Conv2d` layer, `BatchNorm2d` (to mitigate internal covariate shift caused by radiometric variance), a `ReLU` activation, and a `MaxPool2d` for spatial downsampling.
- **Receptive Field Adaptation:** The spatial grid is reduced from 64x64 to an 8x8 dense feature map with 128 channels.
- **Classification Head:** A fully connected network with aggressive Dropout (0.6) to prevent spatial overfitting on highly auto-correlated agricultural and urban textures.

## Data Engineering & Pipeline

Satellite acquisitions with nadir pointing lack a preferential "up/down" orientation and are subject to severe atmospheric and illumination variations. The data pipeline (`src/dataset.py`) handles this via targeted augmentations:
- **Geometric Invariance:** `RandomHorizontalFlip`, `RandomVerticalFlip`, and `RandomRotation(180)` simulate variations in satellite yaw and orbital ground tracks.
- **Radiometric Invariance:** `ColorJitter` simulates solar zenith angle variances and mitigates uncompensated atmospheric scattering (aerosols).
- **Leakage Prevention:** A custom `TransformSubset` wrapper is used to inject augmentations asymmetrically only into the training set, ensuring the validation and test sets remain pristine.

## Performance Metrics

The model was trained using the Adam optimizer with L2 Regularization (Weight Decay: 1e-4) to force distributed feature learning.

| Metric | Value |
| :--- | :--- |
| **Epochs** | 10 |
| **Training Loss** | 0.6553 |
| **Validation Accuracy** | 84.17% |
| **Test Accuracy (Unseen Data)** | 83.63% |

The close alignment between validation and test accuracy demonstrates excellent generalization capabilities with zero overfitting.

## Repository Structure

```text
├── data/                  # EuroSAT dataset (git-ignored)
├── notebooks/             # Exploratory Data Analysis & visual debugging
├── src/
│   ├── dataset.py         # Data loaders, custom Subset wrapper, and augmentations
│   ├── model.py           # EuroSatCNN and CNNBlock architectures
│   └── train.py           # Training loop, validation, and optimal weight checkpointing
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies