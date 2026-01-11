# MNIST Classifier with GPU Support

A complete PyTorch implementation of a Convolutional Neural Network for MNIST digit classification with GPU acceleration.

## Project Structure

```
mnist-classifier/
├── mnist_classifier.py    # Main training script
├── inference.py           # Inference and testing script
├── pyproject.toml         # Project configuration and dependencies
├── data/                 # MNIST dataset (auto-downloaded)
├── best_mnist_model.pth  # Best model checkpoint (after training)
└── final_mnist_model.pth # Final model checkpoint (after training)
```

## Workflow Overview

### 1. Setup Environment

Using `uv` (recommended):

```bash
uv sync
```

This will install all dependencies from `pyproject.toml` and create/update your virtual environment automatically.

For development dependencies:

```bash
uv sync --extra dev
```

Alternatively with `pip`:

```bash
pip install -e .
```

Verify GPU availability:

```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Train the Model

Run the training script:

```bash
uv run python mnist_classifier.py
```

**What happens during training:**

1. **Dataset Loading**: Automatically downloads MNIST dataset (60,000 training images, 10,000 test images)
2. **GPU Detection**: Automatically uses GPU if available, falls back to CPU otherwise
3. **Model Initialization**: Creates CNN with ~100k trainable parameters
4. **Training Loop**: Trains for 10 epochs with the following per epoch:
   - Forward pass on training data
   - Backpropagation and optimization
   - Validation on test set
   - Prints progress every 100 batches
5. **Model Saving**: Saves best model (highest test accuracy) and final model

**Expected output:**
```
Using device: cuda
GPU: NVIDIA GeForce RTX 3080

Loading MNIST dataset...
Training samples: 60000
Test samples: 10000

Starting training...
==================================================

Epoch: 1 [0/60000 (0%)]    Loss: 2.301583
Epoch: 1 [6400/60000 (11%)]    Loss: 0.234567
...

Test set: Average loss: 0.0512, Accuracy: 9837/10000 (98.37%)

✓ Saved new best model with accuracy: 98.37%
```

**Training time:** Approximately 2-3 minutes per epoch on GPU, 15-20 minutes on CPU

### 3. Run Inference

After training, test the model:

```bash
uv run python inference.py
```

This script will:
- Load the best trained model
- Test on 10 random samples from the test set
- Display predictions with confidence scores
- Save visualization to `random_samples_test.png`

### 4. Use Custom Images

To classify your own digit images:

1. Prepare a grayscale image of a digit (any size, will be resized to 28x28)
2. Uncomment the custom image section in `inference.py`
3. Update the `custom_image_path` variable
4. Run inference again

```python
# In inference.py, uncomment:
custom_image_path = 'my_digit.png'
img_tensor, img = preprocess_image(custom_image_path)
prediction, confidence, probabilities = predict(model, img_tensor, device)
```

## Model Architecture

```
MNISTNet(
  Conv2D(1, 32, 3x3) + ReLU + MaxPool(2x2)
  Conv2D(32, 64, 3x3) + ReLU + MaxPool(2x2)
  Flatten
  Linear(3136, 128) + ReLU + Dropout(0.5)
  Linear(128, 10)
)

Total parameters: ~100,000
```

## GPU Optimization Features

The code includes several GPU optimization techniques:

1. **Automatic Device Selection**: Uses GPU if available
2. **Pin Memory**: Faster CPU-to-GPU data transfer
3. **Batch Processing**: Efficient parallel computation
4. **Mixed Precision**: Can be enabled for faster training (requires minor code modification)

## Expected Results

- **Training Accuracy**: ~99.5% after 10 epochs
- **Test Accuracy**: ~98.5-99.0%
- **Training Time**: 
  - GPU (RTX 3080): ~2-3 min/epoch
  - CPU: ~15-20 min/epoch

## Hyperparameters

You can modify these in `mnist_classifier.py`:

```python
batch_size = 64         # Number of samples per batch
epochs = 10             # Number of training epochs
learning_rate = 0.001   # Initial learning rate
```

## Advanced Usage

### Resume Training from Checkpoint

```python
checkpoint = torch.load('best_mnist_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

### Export Model for Production

```python
# Export to TorchScript for deployment
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save('mnist_model_scripted.pt')
```

### Monitor Training with TensorBoard

Add to the training script:

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/mnist_experiment')
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Accuracy/test', test_acc, epoch)
```

Then run: `tensorboard --logdir=runs`

## Troubleshooting

### GPU Not Detected

Check CUDA installation:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory Error

Reduce batch size in the script:
```python
batch_size = 32  # or 16
```

### Slow Training on GPU

Ensure data is pinned to memory and num_workers is set appropriately for your system.

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes, or architectures
2. **Add data augmentation**: Random rotations, translations for better generalization
3. **Try different optimizers**: SGD with momentum, AdamW
4. **Implement early stopping**: Stop training when validation loss stops improving
5. **Export to ONNX**: For deployment on other platforms

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset Info](http://yann.lecun.com/exdb/mnist/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
