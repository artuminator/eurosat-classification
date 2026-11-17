"""
src/dataset.py
Data pipeline per la classificazione LULC su immagini Sentinel-2 (EuroSAT).
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset

class TransformSubset(Dataset):
    """Wrapper per applicare trasformazioni asimmetriche a subset di dati isolati."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_eurosat_dataloaders(data_dir="./data", batch_size=16, val_split=0.15, test_split = 0.15):
    """
    Scarica/carica il dataset EuroSAT e restituisce i DataLoader di train e validazione.
    x_split divide immagini per train (75%) e validazione (15%) e test (15%)
    """
    # Statistiche di normalizzazione ImageNet (baseline standard per backbone RGB)
    eurosat_mean = (0.485, 0.456, 0.406)
    eurosat_std = (0.229, 0.224, 0.225)

    # Pipeline di Training: Augmentation spaziale e radiometrica per EO
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(eurosat_mean, eurosat_std)
    ])

    # Pipeline di Validazione: Streaming puro e standardizzazione
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(eurosat_mean, eurosat_std)
    ])

    # Caricamento del dataset base (scarica in data_dir se non presente)
    # L'output nativo è 64x64 pixel RGB
    base_dataset = datasets.EuroSAT(root=data_dir, download=True, transform=None)

    # Calcolo dei partizionamenti
    total_size = len(base_dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    # Split riproducibile
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(base_dataset, [train_size, val_size, test_size], generator=generator)

    # Iniezione dei transform isolati
    train_dataset = TransformSubset(train_subset, transform=train_transform)
    val_dataset = TransformSubset(val_subset, transform=val_transform)
    test_dataset = TransformSubset(test_subset, transform=val_transform)

    # Istanziazione dei generatori di batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, base_dataset.classes

if __name__ == "__main__":
    print("Inizializzazione della pipeline dati EuroSAT (LULC Sentinel-2)...")
    
    # Esecuzione del modulo in locale (assumendo run da src/)
    # Assicurati che la cartella data/ esista al livello superiore
    train_loader, val_loader, test_loader, classes = get_eurosat_dataloaders(data_dir="../data", batch_size=16, val_split=0.15, test_split=0.15)

    print(f"Classi rilevate ({len(classes)}): {classes}")
    print(f"Batch totali nel Train Loader: {len(train_loader)}")
    print(f"Batch totali nel Val Loader: {len(val_loader)}")
    print(f"Batch totali nel Test Loader: {len(test_loader)}")

    # Estrazione e validazione dimensionale del tensore di input
    images, labels = next(iter(train_loader))
    print(f"\\nShape batch immagini: {images.shape} (Atteso: [16, 3, 64, 64])")
    print(f"Shape batch labels: {labels.shape} (Atteso: [16])")
    print(f"Data type tensore radiometrico: {images.dtype}")
    print("Test pipeline completato con successo.")