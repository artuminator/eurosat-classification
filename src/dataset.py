"""
src/dataset.py
Data pipeline for Land Use and Land Cover (LULC) classification on Sentinel-2 images (EuroSAT).
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset

class TransformSubset(Dataset):
    """
    Wrapper class to apply asymmetric transformations to isolated data subsets.
    This prevents data leakage by ensuring that data augmentation is only applied 
    to the training set and not to the validation or test sets.
    """
    def __init__(self, subset, transform=None):
        # Store the original subset (which contains the data indices)
        self.subset = subset
        # Store the transformation pipeline to be applied
        self.transform = transform

    def __getitem__(self, index):
        # Extract the image (x) and the label (y) from the underlying subset
        x, y = self.subset[index]
        
        # If a transformation pipeline is defined, apply it to the image
        if self.transform:
            x = self.transform(x)
            
        return x, y

    def __len__(self):
        # Return the total number of samples in this specific subset
        return len(self.subset)

def get_eurosat_dataloaders(data_dir="./data", batch_size=16, val_split=0.15, test_split=0.15):
    """
    Downloads/loads the EuroSAT dataset and returns DataLoaders for train, validation, and test.
    The splits parameter divide the images into training (70%), validation (15%), and test (15%).
    """
    # ImageNet normalization statistics (standard baseline for RGB backbones)
    eurosat_mean = (0.485, 0.456, 0.406)
    eurosat_std = (0.229, 0.224, 0.225)

    # Training Pipeline: Spatial and radiometric augmentation tailored for Earth Observation (EO)
    train_transform = transforms.Compose([
        # Geometric Invariance: Satellites have nadir pointing, so there is no fixed "up/down"
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180),
        # Radiometric Invariance: Simulates different solar zenith angles and atmospheric scattering
        transforms.ColorJitter(brightness=0.2),
        # Convert the PIL image to a PyTorch FloatTensor
        transforms.ToTensor(),
        # Standardize the pixel values to help the neural network converge faster
        transforms.Normalize(eurosat_mean, eurosat_std)
    ])

    # Validation/Test Pipeline: Pure streaming and standardization (NO augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(eurosat_mean, eurosat_std)
    ])

    # Load the base dataset (downloads it into data_dir if not found locally)
    # The native output of EuroSAT is 64x64 pixel RGB images
    base_dataset = datasets.EuroSAT(root=data_dir, download=True, transform=None)

    # Calculate the exact number of samples for each partition based on the split ratios
    total_size = len(base_dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    # The remaining samples go to the training set
    train_size = total_size - val_size - test_size

    # Perform a reproducible 3-way split using a fixed random seed
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        base_dataset, 
        [train_size, val_size, test_size], 
        generator=generator
    )

    # Inject the isolated transformations into the respective subsets using our custom wrapper
    train_dataset = TransformSubset(train_subset, transform=train_transform)
    val_dataset = TransformSubset(val_subset, transform=val_transform)
    test_dataset = TransformSubset(test_subset, transform=val_transform)

    # Instantiate the batch generators (DataLoaders)
    # Shuffle is True ONLY for training to ensure stochastic gradient descent works properly
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Return the three DataLoaders and the list of class names
    return train_loader, val_loader, test_loader, base_dataset.classes

if __name__ == "__main__":
    print("Initializing EuroSAT data pipeline (Sentinel-2 LULC)...")
    
    # Execute the module locally (assuming it's run from within the src/ directory)
    # Ensure the data/ folder exists at the parent level
    train_loader, val_loader, test_loader, classes = get_eurosat_dataloaders(
        data_dir="../data", 
        batch_size=16, 
        val_split=0.15, 
        test_split=0.15
    )

    # Print telemetry to verify data partitioning
    print(f"Detected classes ({len(classes)}): {classes}")
    print(f"Total batches in Train Loader: {len(train_loader)}")
    print(f"Total batches in Val Loader: {len(val_loader)}")
    print(f"Total batches in Test Loader: {len(test_loader)}")

    # Extract a single batch to validate the dimensional shape of the input tensor
    images, labels = next(iter(train_loader))
    print(f"\nImage batch shape: {images.shape} (Expected: [16, 3, 64, 64])")
    print(f"Labels batch shape: {labels.shape} (Expected: [16])")
    print(f"Radiometric tensor data type: {images.dtype}")
    print("Pipeline test completed successfully.")
