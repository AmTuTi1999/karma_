"""
Complete Training Pipeline for LSTM and TCN Forecasting Models

This script demonstrates a full training workflow including:
- Model initialization (LSTM/TCN)
- Data loading and preprocessing
- Training with validation
- Checkpointing and evaluation
- Support for multiple metrics
"""

import os
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

# from pathlib import Path
from typing import Dict, Optional, Callable
from datetime import datetime

from models.trainer import Trainer
from models.architectures.lstm_architecture import LSTM
from models.architectures.tcn_architecture import TCN
from utils.types.param_types import LSTMModelParam, TCNModelParam


class TrainingConfig:
    """Configuration for training pipeline"""

    def __init__(self):
        # Model config
        self.model_type: str = "lstm"  # "lstm" or "tcn"
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Training hyperparameters
        self.batch_size: int = 32
        self.learning_rate: float = 0.001
        self.epochs: int = 100
        self.weight_decay: float = 1e-5
        self.clip_grad_norm: float = 1.0

        # Validation and checkpointing
        self.validate_every: int = 1
        self.checkpoint_every: int = 5
        self.best_metric: str = "val_loss"  # metric to track for best model
        self.minimize_metric: bool = True

        # Optimization options
        self.use_amp: bool = False  # mixed precision training
        self.dropout: float = 0.2

        # Scheduler (optional)
        self.use_scheduler: bool = False
        self.scheduler_type: str = "cosine"  # "cosine", "step", "exponential"
        self.scheduler_params: Dict = {}

        # Paths
        self.output_dir: str = "./outputs/training"
        self.checkpoint_dir: str = "./outputs/checkpoints"
        self.results_file: str = "./outputs/results.json"

    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class TrainingPipeline:
    """
    Main training pipeline orchestrator

    Handles model creation, training loop, validation, and results logging
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.model = None
        self.trainer = None
        self.train_history = []
        self.val_history = []

        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def create_model(
        self,
        input_size: int = 10,
        output_size: int = 1,
        forecast_steps: int = 1,
        **kwargs,
    ) -> nn.Module:
        """
        Create LSTM or TCN model based on config

        Args:
            input_size/in_channels: number of input features
            output_size: number of output features per timestep
            forecast_steps: number of future timesteps to predict
            **kwargs: additional model-specific parameters

        Returns:
            Initialized model
        """
        if self.config.model_type.lower() == "lstm":
            params = LSTMModelParam()
            params.input_size = input_size
            params.hidden_size = kwargs.get("hidden_size", 64)
            params.num_layers = kwargs.get("num_layers", 2)
            params.output_size = output_size
            params.forecast_steps = forecast_steps
            params.dropout = self.config.dropout

            self.model = LSTM(params)
            print(f"Created LSTM model: {params.__dict__}")

        elif self.config.model_type.lower() == "tcn":
            params = TCNModelParam()
            params.in_channels = input_size
            params.num_channels = kwargs.get("num_channels", [64, 64, 128])
            params.kernel_size = kwargs.get("kernel_size", 3)
            params.output_size = output_size
            params.forecast_steps = forecast_steps
            params.dropout = self.config.dropout

            self.model = TCN(params)
            print(f"Created TCN model: {params.__dict__}")
        else:
            raise ValueError(
                f"model_type must be 'lstm' or 'tcn', got {self.config.model_type}"
            )

        self.model.to(self.device)
        return self.model

    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for model"""
        if self.model is None:
            raise RuntimeError("Model must be created before setting up optimizer")

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        print(f"Created optimizer: Adam (lr={self.config.learning_rate})")
        return optimizer

    def setup_scheduler(
        self, optimizer: torch.optim.Optimizer, epochs: int
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler (optional)"""
        if not self.config.use_scheduler:
            return None

        if self.config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, **self.config.scheduler_params
            )
        elif self.config.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.scheduler_params.get("step_size", 10),
                gamma=self.config.scheduler_params.get("gamma", 0.1),
            )
        elif self.config.scheduler_type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.config.scheduler_params.get("gamma", 0.95)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

        print(f"Created scheduler: {self.config.scheduler_type}")
        return scheduler

    def setup_trainer(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[Callable] = None,
    ) -> Trainer:
        """Initialize Trainer object"""
        if self.model is None:
            raise RuntimeError("Model must be created before setting up trainer")

        loss_fn = loss_fn or nn.MSELoss()

        self.trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=self.device,
            model_type=self.config.model_type,
            clip_grad_norm=self.config.clip_grad_norm,
            use_amp=self.config.use_amp,
            scheduler=scheduler,
            checkpoint_dir=self.config.checkpoint_dir,
        )
        return self.trainer

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        metric_fns: Optional[Dict[str, Callable]] = None,
    ) -> Dict:
        """
        Execute full training loop

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            metric_fns: Dict of metric_name -> metric_function

        Returns:
            Dictionary with training history
        """
        if self.trainer is None:
            raise RuntimeError("Trainer must be initialized before training")

        print("\n" + "=" * 80)
        print(f"Starting Training: {self.config.model_type.upper()}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print("=" * 80 + "\n")

        start_time = datetime.now()

        for epoch in range(1, self.config.epochs + 1):
            # Training phase
            train_logs = self.trainer.train_epoch(
                train_loader, epoch=epoch, log_interval=50
            )
            self.train_history.append(train_logs)

            print(f"[Epoch {epoch:3d}] Training Loss: {train_logs['train_loss']:.6f}")

            # Validation phase
            if val_loader is not None and epoch % self.config.validate_every == 0:
                val_logs = self.trainer.evaluate(val_loader, metric_fns)
                self.val_history.append({**val_logs, "epoch": epoch})

                print(f"[Epoch {epoch:3d}] Validation Loss: {val_logs['val_loss']:.6f}")
                if metric_fns:
                    for metric_name, metric_val in val_logs.items():
                        if metric_name != "val_loss":
                            print(f"             {metric_name}: {metric_val:.6f}")

            # Checkpointing
            if epoch % self.config.checkpoint_every == 0:
                self.trainer.save_checkpoint(epoch, is_best=False)

            # Step LR scheduler once per epoch (not per batch)
            if self.trainer.scheduler is not None:
                self.trainer.scheduler.step()

            # Best model tracking
            if val_loader is not None and self.config.best_metric in val_logs:
                metric_val = val_logs[self.config.best_metric]
                is_best = False
                if not hasattr(self, "_best_metric_val"):
                    self._best_metric_val = metric_val
                    is_best = True
                elif self.config.minimize_metric and metric_val < self._best_metric_val:
                    self._best_metric_val = metric_val
                    is_best = True
                elif (
                    not self.config.minimize_metric
                    and metric_val > self._best_metric_val
                ):
                    self._best_metric_val = metric_val
                    is_best = True

                if is_best:
                    self.trainer.save_checkpoint(epoch, is_best=True)

        elapsed = datetime.now() - start_time
        print(f"\nTraining completed in {elapsed}")

        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "elapsed_time": str(elapsed),
        }

    def evaluate_final(
        self,
        test_loader: DataLoader,
        metric_fns: Optional[Dict[str, Callable]] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate model on test set with MSE, RMSE, R², and MAE metrics

        Args:
            test_loader: DataLoader for test data
            metric_fns: Dict of metric functions
            checkpoint_path: Path to checkpoint to load (if None, uses current model)

        Returns:
            Dictionary with test metrics including MSE, RMSE, R², MAE
        """
        if checkpoint_path:
            self.trainer.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint: {checkpoint_path}")

        print("\n" + "=" * 80)
        print("Final Evaluation on Test Set")
        print("=" * 80 + "\n")

        # Collect all predictions and ground truth
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                x, y = self.trainer._prepare_batch(batch)
                preds = self.model(x)
                loss = self.trainer.loss_fn(preds, y)

                total_loss += loss.item()
                n_batches += 1

                all_predictions.append(preds.detach().cpu())
                all_targets.append(y.detach().cpu())

        # Concatenate all predictions and targets
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # Compute standard metrics
        mse = torch.mean((predictions - targets) ** 2).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()
        mae = torch.mean(torch.abs(predictions - targets)).item()

        # Compute R²
        ss_res = torch.sum((targets - predictions) ** 2).item()
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Compile results
        results = {
            "val_loss": total_loss / max(1, n_batches),
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

        # Add custom metrics if provided
        if metric_fns:
            metrics = {k: 0.0 for k in metric_fns.keys()}
            for name, fn in metric_fns.items():
                metrics[name] = fn(predictions, targets).item()
            results.update(metrics)

        # Print results
        print(f"Test Loss:  {results['val_loss']:.6f}")
        print(f"MSE:        {results['mse']:.6f}")
        print(f"RMSE:       {results['rmse']:.6f}")
        print(f"MAE:        {results['mae']:.6f}")
        print(f"R²:         {results['r2']:.6f}")

        if metric_fns:
            for name in metric_fns.keys():
                if name not in ["val_loss", "mse", "rmse", "mae", "r2"]:
                    print(f"{name}: {results[name]:.6f}")

        # Save evaluation results to JSON
        eval_results_file = os.path.join(
            self.config.checkpoint_dir, "evaluation_results.json"
        )
        with open(eval_results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nEvaluation results saved to {eval_results_file}")

        return results

    def load_model(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True,
    ) -> nn.Module:
        """
        Load a trained model from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optional optimizer to load state dict
            strict: Whether to strictly enforce that the checkpoint has exactly the same keys

        Returns:
            Loaded model (in eval mode)
        """
        if self.model is None:
            raise RuntimeError(
                "Model must be created before loading checkpoint. "
                "Call create_model() first."
            )

        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"], strict=strict)

        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            print("  Loaded optimizer state")

        epoch = checkpoint.get("epoch", "unknown")
        print(f"  Model loaded from epoch {epoch}")

        self.model.eval()
        return self.model

    def load_pretrained_inference(
        self,
        checkpoint_path: str,
        input_size: int = 10,
        output_size: int = 1,
        forecast_steps: int = 1,
        **kwargs,
    ) -> nn.Module:
        """
        Load a pretrained model for inference (without optimizer/trainer)

        Automatically creates and configures the model architecture
        based on the checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            input_size: Number of input features
            output_size: Number of output features
            forecast_steps: Number of steps to forecast
            **kwargs: Additional architecture parameters

        Returns:
            Loaded model in eval mode, ready for inference
        """
        print("\nLoading pretrained model for inference...")
        print(f"Model type: {self.config.model_type}")

        # Create model architecture
        self.create_model(
            input_size=input_size,
            output_size=output_size,
            forecast_steps=forecast_steps,
            **kwargs,
        )

        # Load checkpoint
        self.load_model(checkpoint_path)

        print(f"Model ready for inference on {self.device}")
        return self.model

    def predict(
        self, data: torch.Tensor, batch_size: int = 32, return_numpy: bool = False
    ) -> torch.Tensor:
        """
        Make predictions with the loaded model

        Args:
            data: Input tensor (batch, seq_len, features) for LSTM or (batch, features, seq_len) for TCN
            batch_size: Batch size for inference
            return_numpy: Whether to return numpy array instead of tensor

        Returns:
            Predictions as tensor or numpy array
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_pretrained_inference() first."
            )
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size].to(self.device)

                # Handle TCN input format if needed
                if self.config.model_type.lower() == "tcn":
                    if batch.ndim == 3:
                        batch = batch.transpose(1, 2).contiguous()

                pred = self.model(batch)
                predictions.append(pred.cpu())

        output = torch.cat(predictions, dim=0)

        if return_numpy:
            return output.numpy()
        return output

    def save_results(self, results: Dict = None):
        """Save training results to JSON"""
        if results is None:
            results = {
                "config": self.config.to_dict(),
                "train_history": self.train_history,
                "val_history": self.val_history,
            }

        with open(self.config.results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {self.config.results_file}")


def example_training_pipeline(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    model_type: str = "lstm",
):
    """
    Example usage of the training pipeline

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        model_type: "lstm" or "tcn"
    """
    # Setup configuration
    config = TrainingConfig()
    config.model_type = model_type
    config.epochs = 100
    config.use_scheduler = True
    config.scheduler_type = "cosine"

    # Initialize pipeline
    pipeline = TrainingPipeline(config)

    # Create model (adjust input_size based on your data)
    pipeline.create_model(
        input_size=2,  # number of features in input
        output_size=2,  # number of features to predict
        forecast_steps=1,  # predict 1 step ahead
        hidden_size=64 if model_type == "lstm" else None,
        num_layers=2 if model_type == "lstm" else None,
        num_channels=[64, 64, 128] if model_type == "tcn" else None,
        kernel_size=3 if model_type == "tcn" else None,
    )

    # Setup optimization
    optimizer = pipeline.setup_optimizer()
    scheduler = pipeline.setup_scheduler(optimizer, config.epochs)
    loss_fn = nn.MSELoss()

    # Initialize trainer
    pipeline.setup_trainer(optimizer, scheduler, loss_fn)

    # Optional: define custom metrics
    def mape(y_pred, y_true):
        """Mean Absolute Percentage Error"""
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8)))

    metric_fns = {"mape": mape}

    # Train
    train_results = pipeline.train(train_loader, val_loader, metric_fns)

    # Evaluate on test set
    if test_loader:
        test_results = pipeline.evaluate_final(
            test_loader,
            metric_fns,
            checkpoint_path=os.path.join(config.checkpoint_dir, "best.pt"),
        )

    # Save results
    pipeline.save_results(test_results if test_loader else train_results)

    return pipeline


if __name__ == "__main__":
    print("Training Pipeline Module Loaded Successfully!")
    print("\nUsage:")
    print("  from training_pipeline import TrainingPipeline, TrainingConfig")
    print("  config = TrainingConfig()")
    print("  pipeline = TrainingPipeline(config)")
    print("  model = pipeline.create_model(...)")
    print("  optimizer = pipeline.setup_optimizer()")
    print("  trainer = pipeline.setup_trainer(optimizer)")
    print("  pipeline.train(train_loader, val_loader)")
