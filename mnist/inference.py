"""
MNIST Model Inference Script
Load a trained model and make predictions on new images
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class MNISTNet(nn.Module):
    """
    Same architecture as in training script
    """
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


def load_model(model_path, device):
    """
    Load a trained model from checkpoint
    """
    model = MNISTNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    print(f"Test accuracy: {checkpoint['accuracy']:.2f}%")
    
    return model


def preprocess_image(image_path):
    """
    Preprocess an image for MNIST model input
    Expects a grayscale image of a digit
    """
    # Load image
    img = Image.open(image_path).convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Apply same transformations as training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    return img_tensor, img


def predict(model, image_tensor, device):
    """
    Make prediction on a single image
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence, probabilities[0].cpu().numpy()


def visualize_prediction(image, prediction, confidence, probabilities):
    """
    Visualize the image and prediction results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Display image
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Predicted: {prediction} (Confidence: {confidence:.2%})')
    ax1.axis('off')
    
    # Display probability distribution
    ax2.bar(range(10), probabilities)
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Probability')
    ax2.set_title('Class Probabilities')
    ax2.set_xticks(range(10))
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to prediction_result.png")
    

def test_on_random_samples(model, device, num_samples=5):
    """
    Test the model on random samples from the MNIST test set
    """
    from torchvision import datasets
    
    # Load MNIST test data
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
    
    # Get random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 3))
    if num_samples == 1:
        axes = [axes]
    
    correct = 0
    
    for idx, sample_idx in enumerate(indices):
        image, label = test_dataset[sample_idx]
        
        # Make prediction
        prediction, confidence, _ = predict(model, image.unsqueeze(0), device)
        
        # Display
        axes[idx].imshow(image.squeeze(), cmap='gray')
        color = 'green' if prediction == label else 'red'
        axes[idx].set_title(
            f'True: {label}\nPred: {prediction}\n({confidence:.1%})',
            color=color
        )
        axes[idx].axis('off')
        
        if prediction == label:
            correct += 1
    
    plt.tight_layout()
    plt.savefig('random_samples_test.png', dpi=150, bbox_inches='tight')
    print(f"\nTested on {num_samples} random samples")
    print(f"Accuracy: {correct}/{num_samples} ({100*correct/num_samples:.1f}%)")
    print("Results saved to random_samples_test.png")


def main():
    """
    Main inference function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    # Load model
    model_path = 'best_mnist_model.pth'
    model = load_model(model_path, device)
    
    print("\n" + "="*50)
    print("Testing on random MNIST samples")
    print("="*50)
    
    # Test on random samples from test set
    test_on_random_samples(model, device, num_samples=10)
    
    # Example of how to use with custom image:
    # Uncomment the following lines and provide your own image
    """
    print("\n" + "="*50)
    print("Testing on custom image")
    print("="*50)
    
    custom_image_path = 'my_digit.png'
    img_tensor, img = preprocess_image(custom_image_path)
    prediction, confidence, probabilities = predict(model, img_tensor, device)
    
    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")
    print("\nAll probabilities:")
    for digit, prob in enumerate(probabilities):
        print(f"  {digit}: {prob:.2%}")
    
    visualize_prediction(img, prediction, confidence, probabilities)
    """


if __name__ == '__main__':
    main()
