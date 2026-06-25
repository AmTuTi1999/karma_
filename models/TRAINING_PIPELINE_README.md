# Training Pipeline Guide

Complete training pipeline for LSTM and TCN time series forecasting models.

## Overview

The training pipeline provides:

- ✅ Easy model creation (LSTM & TCN)
- ✅ Configurable training parameters
- ✅ Automatic checkpointing and best model tracking
- ✅ Mixed precision training support
- ✅ Learning rate scheduling
- ✅ Custom metrics support
- ✅ Comprehensive logging and results tracking
- ✅ Model loading and inference

## Files

1. **training_pipeline.py** - Main training orchestrator
   - `TrainingConfig`: Configuration class for all hyperparameters
   - `TrainingPipeline`: Main pipeline class orchestrating training
   - `example_training_pipeline()`: Complete example function

2. **config_templates.py** - Pre-configured training setups
   - `LSTMConfig`, `TCNConfig`: Architecture-specific configs
   - `FastTrainingConfig`: Quick testing setup
   - `HighAccuracyConfig`: Maximum accuracy configuration
   - `LargeScaleConfig`: GPU-optimized configuration

3. **train_example.py** - Practical examples
   - Demonstrates data loading
   - Shows how to train both LSTM and TCN
   - Includes model comparison

## Quick Start

### Basic Training

```python
from training_pipeline import TrainingPipeline, TrainingConfig
from torch.utils.data import DataLoader

# Create config
config = TrainingConfig()
config.model_type = "lstm"
config.epochs = 100

# Initialize pipeline
pipeline = TrainingPipeline(config)

# Create model
model = pipeline.create_model(
    input_size=10,      # number of input features
    output_size=1,      # number of output features
    forecast_steps=1,   # steps ahead to predict
    hidden_size=64,
    num_layers=2
)

# Setup training
optimizer = pipeline.setup_optimizer()
scheduler = pipeline.setup_scheduler(optimizer, config.epochs)
pipeline.setup_trainer(optimizer, scheduler)

# Train
pipeline.train(train_loader, val_loader)
```

### Using Configuration Templates

```python
from config_templates import get_config
from training_pipeline import TrainingPipeline

# Use pre-configured setup
config = get_config('lstm_default')
pipeline = TrainingPipeline(config)
# ... rest of training code
```

### Training Different Models

#### LSTM Model

```python
config = TrainingConfig()
config.model_type = "lstm"

model = pipeline.create_model(
    input_size=10,
    hidden_size=64,
    num_layers=2,
    output_size=1,
    forecast_steps=1
)
```

#### TCN Model

```python
config = TrainingConfig()
config.model_type = "tcn"

model = pipeline.create_model(
    input_size=10,
    num_channels=[64, 64, 128],
    kernel_size=3,
    output_size=1,
    forecast_steps=1
)
```

## Loading and Inference

### Load a Trained Model for Inference

```python
from training_pipeline import TrainingPipeline, TrainingConfig

# Create pipeline with same config as training
config = TrainingConfig()
config.model_type = "lstm"
pipeline = TrainingPipeline(config)

# Load pretrained model (automatically creates architecture and loads weights)
model = pipeline.load_pretrained_inference(
    checkpoint_path="./outputs/checkpoints/best.pt",
    input_size=10,
    output_size=1,
    forecast_steps=1,
    hidden_size=64,
    num_layers=2
)

# Model is now in eval mode and ready for inference
print(model.training)  # False
```

### Make Predictions

```python
import torch

# Prepare input data (batch_size, seq_len, features) for LSTM
test_data = torch.randn(32, 50, 10)

# Get predictions
predictions = pipeline.predict(test_data, batch_size=8, return_numpy=True)
print(predictions.shape)  # (32, 1)
```

### Load Model with Optimizer State (for Fine-tuning)

```python
# Create and load model with optimizer state
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
pipeline.load_model(
    checkpoint_path="./outputs/checkpoints/best.pt",
    optimizer=optimizer
)

# Continue training with loaded model and optimizer state
pipeline.model.train()
# ... fine-tuning code ...
```

## Configuration Parameters

### Basic Settings

- `model_type`: "lstm" or "tcn"
- `device`: "cuda" or "cpu"
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training

### Optimization

- `learning_rate`: Initial learning rate (default: 0.001)
- `weight_decay`: L2 regularization (default: 1e-5)
- `clip_grad_norm`: Gradient clipping threshold (default: 1.0)
- `dropout`: Dropout rate (default: 0.2)

### Learning Rate Scheduling

- `use_scheduler`: Enable learning rate scheduler (default: False)
- `scheduler_type`: "cosine", "step", or "exponential"
- `scheduler_params`: Scheduler-specific parameters

### Training Setup

- `use_amp`: Mixed precision training (default: False)
- `validate_every`: Validation frequency in epochs (default: 1)
- `checkpoint_every`: Checkpoint frequency in epochs (default: 1)
- `best_metric`: Metric to track for best model (default: "val_loss")
- `minimize_metric`: Whether lower is better (default: True)

### Paths

- `output_dir`: Directory for outputs
- `checkpoint_dir`: Directory for model checkpoints
- `results_file`: Path to results JSON file

## Training Workflow

1. **Create Configuration**

   ```python
   config = TrainingConfig()  # or use a template
   ```

2. **Initialize Pipeline**

   ```python
   pipeline = TrainingPipeline(config)
   ```

3. **Create Model**

   ```python
   model = pipeline.create_model(...)
   ```

4. **Setup Optimization**

   ```python
   optimizer = pipeline.setup_optimizer()
   scheduler = pipeline.setup_scheduler(optimizer, config.epochs)
   ```

5. **Setup Trainer**

   ```python
   pipeline.setup_trainer(optimizer, scheduler)
   ```

6. **Train Model**

   ```python
   pipeline.train(train_loader, val_loader, metric_fns)
   ```

7. **Evaluate on Test Set**

   ```python
   results = pipeline.evaluate_final(
       test_loader,
       metric_fns,
       checkpoint_path="./outputs/checkpoints/best.pt"
   )
   ```

8. **Save Results**
   ```python
   pipeline.save_results()
   ```

## Custom Metrics

Add custom metrics to track during training:

```python
def mape(y_pred, y_true):
    """Mean Absolute Percentage Error"""
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8)))

def rmse(y_pred, y_true):
    """Root Mean Squared Error"""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

metric_fns = {
    'mape': mape,
    'rmse': rmse
}

pipeline.train(train_loader, val_loader, metric_fns)
```

## Advanced Configuration

### Mixed Precision Training

```python
config.use_amp = True  # Faster training on modern GPUs
```

### Learning Rate Scheduling

```python
config.use_scheduler = True
config.scheduler_type = "cosine"

# Or with step decay
config.scheduler_type = "step"
config.scheduler_params = {
    "step_size": 30,
    "gamma": 0.1
}
```

### Gradient Clipping

```python
config.clip_grad_norm = 1.0  # Clip gradients to max norm
```

## Examples

### Complete Example with Both Models

```python
from train_example import compare_models
from torch.utils.data import DataLoader

# Your data loaders
train_loader = DataLoader(...)
val_loader = DataLoader(...)
test_loader = DataLoader(...)

# Train both models and compare
lstm_pipeline, tcn_pipeline = compare_models(test_loader)
```

### Training with Custom Configuration

```python
from config_templates import HighAccuracyConfig
from training_pipeline import TrainingPipeline

config = HighAccuracyConfig()
pipeline = TrainingPipeline(config)

# Rest of training...
```

### Train Model and Load for Inference

```python
from train_example import train_and_load_example

# Trains a model and then loads it for inference in one go
pipeline, inference_pipeline, predictions = train_and_load_example()
```

### Load Existing Checkpoint and Make Predictions

```python
from train_example import load_and_inference_lstm
import os

# Load a previously trained model
checkpoint_path = "./outputs/checkpoints/best.pt"

if os.path.exists(checkpoint_path):
    pipeline, predictions = load_and_inference_lstm(
        checkpoint_path=checkpoint_path,
        n_samples=10,
        seq_length=50,
        n_features=10
    )
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions:\n{predictions}")
```

### Command Line Usage

```bash
# Train a new model
python train_example.py --mode train --model lstm --epochs 50

# Load an existing model for inference
python train_example.py --mode load --model lstm --checkpoint ./outputs/checkpoints/best.pt

# Train a model and immediately load it for inference
python train_example.py --mode train_and_load --model lstm --epochs 50
```

## Output Structure

After training, the following directories are created:

```
outputs/
├── training/          # Training outputs
│   └── results.json   # Training history and metrics
└── checkpoints/       # Model checkpoints
    ├── checkpoint_epoch5.pt
    ├── checkpoint_epoch10.pt
    └── best.pt        # Best model based on configured metric
```

## Results JSON Format

```json
{
  "config": {
    "model_type": "lstm",
    "epochs": 100,
    "learning_rate": 0.001,
    ...
  },
  "train_history": [
    {"train_loss": 0.5234},
    {"train_loss": 0.4521},
    ...
  ],
  "val_history": [
    {"epoch": 1, "val_loss": 0.6123, "mape": 0.0432},
    {"epoch": 2, "val_loss": 0.5891, "mape": 0.0398},
    ...
  ]
}
```

## Performance Tips

1. **Batch Size**: Larger batches (64-128) for faster training, smaller (16-32) for stability
2. **Learning Rate**: Start with 0.001, adjust based on validation loss
3. **Early Stopping**: Monitor `val_loss` and stop if not improving
4. **Mixed Precision**: Enable `use_amp=True` for 2x speedup on modern GPUs
5. **Gradient Clipping**: Use for stability with recurrent models (LSTM)
6. **Learning Rate Schedule**: Cosine annealing often works better than fixed LR

## Troubleshooting

### Model not converging

- Reduce learning rate
- Increase batch size
- Enable gradient clipping
- Check data normalization

### Out of Memory (OOM)

- Reduce batch size
- Reduce model hidden size
- Enable gradient checkpointing
- Use mixed precision (use_amp=True)

### Validation loss high compared to training

- Add dropout (increase dropout rate)
- Add weight decay
- Reduce model complexity
- Use early stopping

## Architecture Comparison

### LSTM

- Better for capturing long-term dependencies
- Slower inference
- More stable training
- Better for irregular time series

### TCN

- Faster inference (parallel)
- Better for fixed-size context windows
- Requires careful hyperparameter tuning
- Better for very long sequences

## References

- LSTM: [Hochreiter & Schmidhuber (1997)](https://pubmed.ncbi.nlm.nih.gov/9377276/)
- TCN: [Bai et al. (2018)](https://arxiv.org/abs/1803.01271)
