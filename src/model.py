"""
src/model.py
CNN architecture for Land Use and Land Cover (LULC) classification on Sentinel-2 images (EuroSAT).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CNNBlock(nn.Module):
    """
    Modular convolutional block: handles feature extraction, radiometric normalization, 
    non-linear activation, and spatial downsampling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CNNBlock, self).__init__()

        # Sequential container for the building blocks of the convolutional stage
        self.block = nn.Sequential(
            # 2D Convolution: extracts spatial features (edges, textures) from the satellite patch
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            # Batch Normalization: mitigates internal covariate shift caused by radiometric variance 
            # (e.g., changes in illumination or atmospheric conditions across batches)
            nn.BatchNorm2d(out_channels),
            # ReLU Activation: introduces non-linearity to the network
            nn.ReLU(),
            # Max Pooling: reduces spatial dimensions by a factor of 2, expanding the receptive field
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """Forward pass for the CNNBlock."""
        return self.block(x)


class EuroSatCNN(nn.Module):
    """
    CNN architecture adapted for 64x64 pixel optical payloads (10m Ground Sample Distance).
    """
    def __init__(self, num_classes=10):
        super(EuroSatCNN, self).__init__()

        # --- Feature Extractor ---
        # Successive blocks extract increasingly high-level semantic representations of the landscape
        self.conv_block1 = CNNBlock(in_channels=3, out_channels=32)
        self.conv_block2 = CNNBlock(in_channels=32, out_channels=64)
        self.conv_block3 = CNNBlock(in_channels=64, out_channels=128)

        # --- Classifier Head ---
        # The initial spatial grid is 64x64. 
        # After 3 MaxPool operations (stride 2), the grid is reduced by a factor of 2^3 = 8.
        # The resulting tensor shape from conv_block3 is [Batch, 128 channels, 8, 8].
        # Flattening yields: 128 * 8 * 8 = 8192 dense features.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8192, out_features=512),
            nn.ReLU(),
            # Aggressive Dropout (0.6): prevents spatial overfitting on highly auto-correlated 
            # EO textures (like forests or agricultural fields), forcing robust, distributed learning.
            nn.Dropout(0.6),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        """Defines the forward pass through the feature extractor and the classifier."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":
    # Local debugging and dimensional testing
    print("Initializing EuroSatCNN model...")

    # Instantiate the model for the 10 classes of the EuroSAT dataset
    model = EuroSatCNN(num_classes=10)

    # Create a dummy tensor simulating a batch of Sentinel-2 MSI optical patches
    # Dimensions: [Batch_Size=16, Channels=3 (RGB), Height=64, Width=64]
    dummy_input = torch.randn(16, 3, 64, 64)
    print(f"Input tensor shape: {dummy_input.shape}")

    # Forward pass to verify tensor shape congruence through the network
    output = model(dummy_input)
    print(f"Output tensor shape: {output.shape} (Expected: [16, 10])")
    print("Architecture dimensional test passed successfully.")
