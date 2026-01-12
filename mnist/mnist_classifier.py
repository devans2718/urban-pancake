"""
MNIST Digit Classifier with GPU Support
This script trains a convolutional neural network on the MNIST dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from pathlib import Path

from common.paths import OUTPUT, ensure_directory


class MNISTNet(nn.Module):
    """
    Convolutional Neural Network for MNIST classification
    Architecture: Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> FC -> Dropout -> FC
    """

    def __init__(self):
        super(MNISTNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First conv block
        x = self.pool(torch.relu(self.conv1(x)))  # 28x28 -> 14x14

        # Second conv block
        x = self.pool(torch.relu(self.conv2(x)))  # 14x14 -> 7x7

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def get_data_loaders(batch_size=64):
    """
    Prepare MNIST data loaders with data augmentation
    """
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True  # Faster data transfer to GPU
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to GPU
        data, target = data.to(device), target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def test(model, device, test_loader, criterion):
    """
    Evaluate the model on test data
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Move data to GPU
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            test_loss += criterion(output, target).item()

            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return test_loss, accuracy


def main():
    """
    Main training loop
    """

    # Ensure output directory exists
    ensure_directory(OUTPUT / 'mnist')

    # Hyperparameters
    batch_size = 64
    epochs = 10
    learning_rate = 0.001

    # Set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(
            f'Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB')

    # Get data loaders
    print('\nLoading MNIST dataset...')
    train_loader, test_loader = get_data_loaders(batch_size)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')

    # Initialize model
    model = MNISTNet().to(device)
    print(f'\nModel architecture:\n{model}')

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    print('\n' + '='*50)
    print('Starting training...')
    print('='*50 + '\n')

    best_accuracy = 0.0
    train_history = []
    test_history = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )

        # Test
        test_loss, test_acc = test(model, device, test_loader, criterion)

        # Update learning rate
        scheduler.step()

        # Save history
        train_history.append({'loss': train_loss, 'accuracy': train_acc})
        test_history.append({'loss': test_loss, 'accuracy': test_acc})

        epoch_time = time.time() - start_time

        print(f'Epoch {epoch} completed in {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50 + '\n')

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
                'loss': test_loss,
            }, OUTPUT / 'mnist/best_mnist_model.pth')
            print(f'✓ Saved new best model with accuracy: {test_acc:.2f}%\n')

    print('='*50)
    print('Training completed!')
    print(f'Best test accuracy: {best_accuracy:.2f}%')
    print('='*50)

    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_history': train_history,
        'test_history': test_history,
    }, OUTPUT / 'mnist/final_mnist_model.pth')

    print('\nModel saved to mnist/outputs/best_mnist_model.pth and mnist/outputs/final_mnist_model.pth')


if __name__ == '__main__':
    main()
