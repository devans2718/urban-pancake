"""
Training Visualization Script
Analyze and visualize model training history and performance
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import seaborn as sns


def plot_training_history(checkpoint_path='final_mnist_model.pth'):
    """
    Plot training and validation loss/accuracy over epochs
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'train_history' not in checkpoint or 'test_history' not in checkpoint:
        print("No training history found in checkpoint")
        return
    
    train_history = checkpoint['train_history']
    test_history = checkpoint['test_history']
    epochs = range(1, len(train_history) + 1)
    
    # Extract metrics
    train_loss = [h['loss'] for h in train_history]
    train_acc = [h['accuracy'] for h in train_history]
    test_loss = [h['loss'] for h in test_history]
    test_acc = [h['accuracy'] for h in test_history]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, test_loss, 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, train_acc, 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, test_acc, 'r-s', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Training history plot saved to training_history.png")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Total Epochs: {len(train_history)}")
    print(f"\nFinal Training Accuracy: {train_acc[-1]:.2f}%")
    print(f"Final Validation Accuracy: {test_acc[-1]:.2f}%")
    print(f"\nBest Validation Accuracy: {max(test_acc):.2f}% (Epoch {test_acc.index(max(test_acc)) + 1})")
    print(f"Final Training Loss: {train_loss[-1]:.4f}")
    print(f"Final Validation Loss: {test_loss[-1]:.4f}")
    print("="*50 + "\n")


def plot_confusion_matrix(model_path='best_mnist_model.pth'):
    """
    Generate and plot confusion matrix on test set
    """
    from sklearn.metrics import confusion_matrix
    import torch.nn as nn
    
    # Model definition (same as training script)
    class MNISTNet(nn.Module):
        def __init__(self):
            super(MNISTNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False
    )
    
    # Get predictions
    all_preds = []
    all_labels = []
    
    print("Generating predictions for confusion matrix...")
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - MNIST Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix saved to confusion_matrix.png")
    
    # Calculate per-class accuracy
    print("\n" + "="*50)
    print("PER-CLASS ACCURACY")
    print("="*50)
    
    for i in range(10):
        class_correct = cm[i, i]
        class_total = cm[i, :].sum()
        accuracy = 100 * class_correct / class_total
        print(f"Digit {i}: {class_correct}/{class_total} ({accuracy:.2f}%)")
    
    print("="*50 + "\n")


def visualize_filters(model_path='best_mnist_model.pth'):
    """
    Visualize learned convolutional filters
    """
    import torch.nn as nn
    
    class MNISTNet(nn.Module):
        def __init__(self):
            super(MNISTNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get first conv layer filters
    filters = model.conv1.weight.data.cpu().numpy()
    
    # Plot first 32 filters
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[0]:
            ax.imshow(filters[i, 0], cmap='gray')
            ax.set_title(f'Filter {i+1}')
        ax.axis('off')
    
    plt.suptitle('First Convolutional Layer Filters (3x3)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('conv_filters.png', dpi=150, bbox_inches='tight')
    print("Convolutional filters visualization saved to conv_filters.png")


def analyze_misclassifications(model_path='best_mnist_model.pth', num_examples=20):
    """
    Find and visualize misclassified examples
    """
    import torch.nn as nn
    
    class MNISTNet(nn.Module):
        def __init__(self):
            super(MNISTNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Find misclassifications
    misclassified = []
    
    print("Finding misclassified examples...")
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, label = test_dataset[idx]
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            
            if pred != label:
                confidence = torch.softmax(output, dim=1)[0][pred].item()
                misclassified.append({
                    'image': image,
                    'true_label': label,
                    'pred_label': pred,
                    'confidence': confidence
                })
                
                if len(misclassified) >= num_examples:
                    break
    
    # Plot misclassified examples
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(misclassified):
            item = misclassified[idx]
            ax.imshow(item['image'].squeeze(), cmap='gray')
            ax.set_title(f"True: {item['true_label']}, Pred: {item['pred_label']}\n"
                        f"Conf: {item['confidence']:.2%}",
                        color='red')
        ax.axis('off')
    
    plt.suptitle('Misclassified Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('misclassifications.png', dpi=150, bbox_inches='tight')
    print(f"Found {len(misclassified)} misclassified examples")
    print("Misclassifications visualization saved to misclassifications.png\n")


def main():
    """
    Run all visualization and analysis
    """
    print("\n" + "="*60)
    print("MNIST MODEL ANALYSIS AND VISUALIZATION")
    print("="*60 + "\n")
    
    # 1. Plot training history
    print("1. Plotting training history...")
    plot_training_history()
    
    # 2. Generate confusion matrix
    print("\n2. Generating confusion matrix...")
    plot_confusion_matrix()
    
    # 3. Visualize filters
    print("\n3. Visualizing convolutional filters...")
    visualize_filters()
    
    # 4. Analyze misclassifications
    print("\n4. Analyzing misclassifications...")
    analyze_misclassifications()
    
    print("\n" + "="*60)
    print("All visualizations completed!")
    print("Generated files:")
    print("  - training_history.png")
    print("  - confusion_matrix.png")
    print("  - conv_filters.png")
    print("  - misclassifications.png")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
