import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):

        super(CNNBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):

        return self.block(x)


class EuroSatCNN(nn.Module):

    def __init__(self, num_classes=10):

        super(EuroSatCNN, self).__init__()

        self.conv_block1 = CNNBlock(in_channels=3, out_channels=32)
        self.conv_block2 = CNNBlock(in_channels=32, out_channels=64)
        self.conv_block3 = CNNBlock(in_channels=64, out_channels=128)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8192, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=512, out_features=num_classes)
        )


    def forward(self, x):

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    print("Inizializzazione del modello EuroSatCNN...")

    model = EuroSatCNN(num_classes=10)

    dummy_input = torch.randn(16, 3, 64, 64)
    print(f"Shape tensore di input: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Shape tensore di output: {output.shape} (Atteso: [16,10])")
    print("Test dimensionale superato con successo")