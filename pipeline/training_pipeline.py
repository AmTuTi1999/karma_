"""
Unified training pipeline for all datasets and model architectures.

Usage:
    python -m pipeline.training_pipeline --dataset etth1 --model lstm
    python -m pipeline.training_pipeline --dataset weather --model both --epochs 100
    python -m pipeline.training_pipeline --dataset exchange_rate --model tcn --seed 0
"""

import argparse
import importlib
from pathlib import Path

import torch
import yaml

from models.training_pipeline import TrainingConfig, TrainingPipeline

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
DATASETS = [
    "etth1", "ettm2", "weather", "exchange_rate",
    "beijing", "electricity", "web_traffic",
]


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_model(
    dataset_name: str,
    model_type: str,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple:
    """
    Train a single model on a dataset using configs from YAML files.

    Parameters
    ----------
    dataset_name : one of DATASETS
    model_type   : 'lstm' or 'tcn'
    epochs       : training epochs (overrides training/default.yaml)
    batch_size   : mini-batch size
    seed         : random seed

    Returns
    -------
    (pipeline, test_results)
    """
    dataset_cfg = _load_yaml(CONFIGS_DIR / "datasets" / f"{dataset_name}.yaml")
    model_cfg = _load_yaml(CONFIGS_DIR / "models" / f"{model_type}.yaml")
    train_cfg = _load_yaml(CONFIGS_DIR / "training" / "default.yaml")

    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"TRAINING {model_type.upper()} — {dataset_name}")
    print("=" * 60)

    mod = importlib.import_module(dataset_cfg["dataset_module"])
    train_loader, val_loader, test_loader = mod.get_dataloaders(
        data_dir=dataset_cfg["data_dir"],
        batch_size=batch_size,
    )

    # Infer D from data when not fixed in config
    D = dataset_cfg.get("D")
    if D is None:
        sample_x, _ = next(iter(train_loader))
        D = sample_x.shape[-1]

    forecast_steps = dataset_cfg.get("forecast_steps", 1)

    config = TrainingConfig()
    config.model_type = model_type
    config.epochs = epochs
    config.batch_size = batch_size
    config.learning_rate = train_cfg["learning_rate"]
    config.use_scheduler = train_cfg.get("use_scheduler", True)
    config.scheduler_type = train_cfg.get("scheduler_type", "cosine")
    config.dropout = train_cfg.get("dropout", 0.2)
    config.checkpoint_dir = dataset_cfg["checkpoints"][model_type]
    if model_type == "tcn":
        config.clip_grad_norm = train_cfg.get("clip_grad_norm", 1.0)

    pipeline = TrainingPipeline(config)

    model_kwargs = dict(input_size=D, output_size=D, forecast_steps=forecast_steps)
    if model_type == "lstm":
        model_kwargs.update(
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
        )
    else:
        model_kwargs.update(
            num_channels=model_cfg["num_channels"],
            kernel_size=model_cfg["kernel_size"],
        )

    pipeline.create_model(**model_kwargs)
    optimizer = pipeline.setup_optimizer()
    scheduler = pipeline.setup_scheduler(optimizer, epochs)
    pipeline.setup_trainer(optimizer, scheduler)
    pipeline.train(train_loader, val_loader)

    test_results = pipeline.evaluate_final(
        test_loader,
        checkpoint_path=f"{config.checkpoint_dir}/best.pt",
    )

    print(f"\nCheckpoint saved: {config.checkpoint_dir}/best.pt")
    return pipeline, test_results


def main():
    p = argparse.ArgumentParser(description="Train time series forecasting models")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--model", default="both", choices=["lstm", "tcn", "both"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    models = ["lstm", "tcn"] if args.model == "both" else [args.model]
    for m in models:
        train_model(
            dataset_name=args.dataset,
            model_type=m,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
