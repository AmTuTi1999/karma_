import torch
import numpy as np
from models.training_pipeline import TrainingPipeline, TrainingConfig


def load_pretrained_oracle(
    checkpoint_path: str,
    model_type: str = "lstm",
    input_size: int = 10,
    output_size: int = 1,
    forecast_steps: int = 1,
    batch_size: int = 256,
    **kwargs,
):
    """
    Load a pretrained model from training pipeline as oracle.
    The model takes window of shape (W, D) and predicts (D,) continuous values.

    Parameters
    ----------
    batch_size : GPU mini-batch size used during inference. Larger values
                 improve throughput on GPU at the cost of peak VRAM usage.
                 Set to None to run the entire input as one batch.
    """

    config = TrainingConfig()
    config.model_type = model_type
    pipeline = TrainingPipeline(config)
    pipeline.load_pretrained_inference(
        checkpoint_path,
        input_size=input_size,
        output_size=output_size,
        forecast_steps=forecast_steps,
        **kwargs,
    )

    _batch_size = batch_size

    def f_oracle(window: np.ndarray) -> np.ndarray:
        """
        window : (W, D)    → returns (forecast_steps, D)
                 (B, W, D) → returns (B, forecast_steps, D)
        """
        is_single = window.ndim == 2
        window_tensor = (
            torch.from_numpy(window).float().unsqueeze(0)
            if is_single
            else torch.from_numpy(window).float()
        )
        bs = _batch_size if _batch_size is not None else len(window_tensor)
        pred = pipeline.predict(window_tensor, batch_size=bs, return_numpy=True)
        return pred.squeeze(0) if is_single else pred

    return f_oracle


def _make_oracle(pretrained_model):
    """
    Wrap pretrained model into oracle interface.
    (W, D)    → (forecast_steps, D)
    (B, W, D) → (B, forecast_steps, D)
    """

    def f_oracle(window: np.ndarray) -> np.ndarray:
        is_single = window.ndim == 2
        window_tensor = (
            torch.from_numpy(window).float().unsqueeze(0)
            if is_single
            else torch.from_numpy(window).float()
        )
        pred = pretrained_model.predict(
            window_tensor, batch_size=1, return_numpy=True
        )  # (1, forecast_steps, D) or (B, forecast_steps, D)
        return pred.squeeze(0) if is_single else pred

    return f_oracle
