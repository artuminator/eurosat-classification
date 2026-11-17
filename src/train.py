"""
src/train.py
Training pipeline for the EuroSatCNN LULC model with telemetry on Train, Validation, and Test sets.
"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import get_eurosat_dataloaders
from model import EuroSatCNN

def train_epoch(model, train_loader, loss_function, optimizer, device):
    """Executes a single training epoch over the provided dataset."""
    
    # Set the model to training mode (enables layers like Dropout and updates BatchNorm statistics)
    model.train()
    running_loss = 0.0

    # Iterate over batches of images and their corresponding ground-truth labels
    for images, labels in train_loader:
        # Move the data to the target compute device (CPU or GPU)
        images, labels = images.to(device), labels.to(device)

        # 1. Clear previous gradients to prevent accumulation
        optimizer.zero_grad()
        
        # 2. Forward pass: compute the model's predictions for the current batch
        outputs = model(images)
        
        # 3. Calculate the loss between predictions and actual labels
        loss = loss_function(outputs, labels)
        
        # 4. Backward pass: compute the gradients of the loss with respect to model parameters
        loss.backward()
        
        # 5. Optimizer step: update the model's weights
        optimizer.step()

        # Accumulate the total loss for this batch (scaled by the batch size)
        running_loss += loss.item() * images.size(0)

    # Calculate the average loss over the entire training dataset
    epoch_loss = running_loss / len(train_loader.dataset)
    
    return epoch_loss


def validate_epoch(model, val_loader, loss_function, device):
    """Executes a validation/test loop on unseen data to evaluate generalization."""
    
    # Set the model to evaluation mode (disables Dropout, freezes BatchNorm statistics)
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient computation to save memory and simulate real-world inference
    with torch.no_grad():

        for images, labels in val_loader:
            # Move tensors to the designated hardware device
            images, labels = images.to(device), labels.to(device)

            # Forward pass: extract features and classify
            outputs = model(images)
            
            # Compute the validation loss
            val_loss = loss_function(outputs, labels)
            running_val_loss += val_loss.item() * images.size(0)

            # Identify the predicted class (the index with the highest logit score)
            _, predicted = torch.max(outputs, 1)

            # Track total samples and correct predictions for accuracy calculation
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute average loss and accuracy percentage
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_accuracy = 100.0 * correct / total

    return epoch_val_loss, epoch_accuracy


def training_loop(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device):
    """
    Trains and validates a PyTorch neural network model.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        loss_function (callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        num_epochs (int): The total number of epochs to train for.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to run training on.

    Returns:
        tuple: A tuple containing the best trained model and a list of metrics
               (train_losses, val_losses, val_accuracies).
    """
    # Move the model to the specified device
    model.to(device)
    
    # Initialize variables to track the best performing model state
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Initialize lists to store telemetry data (useful for plotting/debugging later)
    train_losses, val_losses, val_accuracies = [], [], []
    
    print("--- Training Started ---")
    
    # Loop over the dataset multiple times (epochs)
    for epoch in range(num_epochs):
        # Execute one epoch of training and record the loss
        epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
        train_losses.append(epoch_loss)
        
        # Execute one epoch of validation and record the metrics
        epoch_val_loss, epoch_accuracy = validate_epoch(model, val_loader, loss_function, device)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)
        
        # Output the epoch's telemetry
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.2f}%")
        
        # Model Checkpointing: save the model's weights ONLY if validation accuracy improves.
        # This acts as a manual Early Stopping mechanism to prevent keeping an overfitted model.
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            # Create a deep copy of the model's state dictionary to preserve the optimal weights
            best_model_state = copy.deepcopy(model.state_dict())
            
    print("--- Finished Training ---")
    
    # Restore the optimal weights before returning the model
    if best_model_state:
        print(f"\n--- Returning best model with {best_val_accuracy:.2f}% validation accuracy, achieved at epoch {best_epoch} ---")
        model.load_state_dict(best_model_state)
    
    # Consolidate all metrics into a single list
    metrics = [train_losses, val_losses, val_accuracies]
    
    # Return the best trained model and its historical metrics
    return model, metrics


if __name__ == "__main__":
    # Setup the compute device (utilizes NVIDIA GPU if available for accelerated training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target hardware: {device}\n")

    # 1. Initialize the dataloaders for the 3-way split (Train, Validation, Test)
    train_loader, val_loader, test_loader, classes = get_eurosat_dataloaders(
        data_dir="../data", batch_size=16, val_split=0.15, test_split=0.15
    )

    # 2. Initialize the CNN architecture dynamically based on the detected classes
    model = EuroSatCNN(num_classes=len(classes))

    # 3. Define the loss function (Cross-Entropy for multi-class classification) 
    # and the Adam optimizer with Weight Decay (L2 Regularization) to penalize large weights 
    # and force the network to learn distributed, generalizable features.
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 4. Launch the training loop over 10 epochs
    trained_model, training_metrics = training_loop(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        loss_function=loss_function, 
        optimizer=optimizer, 
        num_epochs=10, 
        device=device
    )

    # 5. Final Unbiased Evaluation: test the optimized model on completely unseen Earth Observation data
    print("\n--- Running inference on the Test Set (Flight Simulation) ---")
    test_loss, test_acc = validate_epoch(trained_model, test_loader, loss_function, device)
    print(f"Test Set Performance: Loss = {test_loss:.4f}, Accuracy = {test_acc:.2f}%")
