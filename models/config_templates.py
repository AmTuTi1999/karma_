"""
Training Configuration Templates

Provides pre-configured training setups for different scenarios.
Modify these configs for your specific use case.
"""

from training_pipeline import TrainingConfig


class LSTMConfig(TrainingConfig):
    """Configuration optimized for LSTM models"""

    def __init__(self):
        super().__init__()
        self.model_type = "lstm"

        # LSTM-specific hyperparameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 100
        self.dropout = 0.3
        self.clip_grad_norm = 1.0

        # Scheduler
        self.use_scheduler = True
        self.scheduler_type = "cosine"

        # Model architecture
        self.lstm_hidden_size = 128
        self.lstm_num_layers = 2


class TCNConfig(TrainingConfig):
    """Configuration optimized for TCN models"""

    def __init__(self):
        super().__init__()
        self.model_type = "tcn"

        # TCN-specific hyperparameters
        self.batch_size = 32
        self.learning_rate = 0.0005
        self.epochs = 100
        self.dropout = 0.2
        self.clip_grad_norm = 1.0
        self.use_amp = False

        # Scheduler
        self.use_scheduler = True
        self.scheduler_type = "step"
        self.scheduler_params = {"step_size": 20, "gamma": 0.5}

        # Model architecture
        self.tcn_num_channels = [64, 64, 128]
        self.tcn_kernel_size = 3


class FastTrainingConfig(TrainingConfig):
    """Quick training for testing and debugging"""

    def __init__(self):
        super().__init__()
        self.model_type = "lstm"

        self.batch_size = 64
        self.learning_rate = 0.001
        self.epochs = 10
        self.dropout = 0.1
        self.validate_every = 1
        self.checkpoint_every = 5
        self.use_scheduler = False
        self.use_amp = True


class HighAccuracyConfig(TrainingConfig):
    """Configuration for maximum accuracy (slower training)"""

    def __init__(self):
        super().__init__()
        self.model_type = "lstm"

        self.batch_size = 16
        self.learning_rate = 0.0001
        self.epochs = 200
        self.weight_decay = 1e-4
        self.dropout = 0.4
        self.clip_grad_norm = 0.5

        self.validate_every = 1
        self.checkpoint_every = 5

        # Strong regularization
        self.use_scheduler = True
        self.scheduler_type = "cosine"
        self.use_amp = False


class LargeScaleConfig(TrainingConfig):
    """Configuration for large-scale training on GPUs"""

    def __init__(self):
        super().__init__()
        self.model_type = "tcn"

        self.batch_size = 128
        self.learning_rate = 0.001
        self.epochs = 150
        self.dropout = 0.2

        # Use mixed precision for faster training
        self.use_amp = True
        self.clip_grad_norm = 1.0

        self.validate_every = 2
        self.checkpoint_every = 10

        self.use_scheduler = True
        self.scheduler_type = "cosine"


# Example configurations by use case
CONFIGS = {
    "lstm_default": LSTMConfig,
    "tcn_default": TCNConfig,
    "fast": FastTrainingConfig,
    "accuracy": HighAccuracyConfig,
    "large_scale": LargeScaleConfig,
}


def get_config(name: str) -> TrainingConfig:
    """
    Get a pre-configured training config by name

    Args:
        name: Configuration name ('lstm_default', 'tcn_default', 'fast', 'accuracy', 'large_scale')

    Returns:
        TrainingConfig instance
    """
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]()


def print_available_configs():
    """Print all available configuration templates"""
    print("\nAvailable Training Configurations:")
    print("-" * 60)
    for name, config_class in CONFIGS.items():
        config = config_class()
        print(f"\n{name}:")
        print(f"  Model Type: {config.model_type}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Dropout: {config.dropout}")
        print(
            f"  Scheduler: {config.scheduler_type if config.use_scheduler else 'Off'}"
        )
        print(f"  Mixed Precision: {'On' if config.use_amp else 'Off'}")


if __name__ == "__main__":
    print_available_configs()
