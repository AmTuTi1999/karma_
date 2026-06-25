import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Optional, Dict


class Trainer:
    """
    Generic trainer for regression forecasting models (LSTMForecast or TCN).

    Args:
      model: nn.Module
      optimizer: torch.optim.Optimizer
      loss_fn: loss function (e.g., nn.MSELoss())
      device: torch.device or "cuda"/"cpu"
      model_type: "lstm" or "tcn"  (affects input shape handling)
      clip_grad_norm: float or None
      use_amp: bool use mixed precision
      scheduler: optional LR scheduler
      checkpoint_dir: dir to save checkpoints
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: Optional[torch.device] = None,
        model_type: str = "lstm",
        clip_grad_norm: Optional[float] = 1.0,
        use_amp: bool = False,
        scheduler: Optional[object] = None,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = (
            torch.device(
                "cuda"
                if torch.cuda.is_available()
                and (device is None or device.type == "cuda")
                else "cpu"
            )
            if device is None
            else device
        )
        self.model_type = model_type.lower()
        assert self.model_type in ("lstm", "tcn"), "model_type must be 'lstm' or 'tcn'"
        self.clip_grad_norm = clip_grad_norm
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.to(self.device)

    def _prepare_batch(self, batch):
        """
        Expects batch from DataLoader to be (x, y) where:
        - x: (batch, seq_length, features) for all models
        - y: (batch, forecast_steps, features) or (batch, forecast_steps)

        This function will:
        - Move tensors to device
        - Transpose x to (batch, features, seq_length) for TCN
        - Keep x as (batch, seq_length, features) for LSTM
        """
        if isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            x, y = batch

        x = x.to(self.device)
        y = y.to(self.device)

        if self.model_type == "tcn":
            # TCN expects (batch, channels, seq_length)
            # Transpose from (batch, seq_length, features) to (batch, features, seq_length)
            if x.ndim == 3:
                x = x.transpose(1, 2).contiguous()
            elif x.ndim == 2:
                # (batch, seq_length) -> add channel dim -> (batch, 1, seq_length)
                x = x.unsqueeze(1)
        # else: LSTM expects (batch, seq_length, features) which is the default

        return x, y

    def train_epoch(
        self, train_loader: DataLoader, epoch: int = 0, log_interval: int = 100
    ) -> Dict:
        self.model.train()
        running_loss = 0.0
        n_batches = 0
        start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            x, y = self._prepare_batch(batch)
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    preds = self.model(x)
                    loss = self.loss_fn(preds, y)
                self.scaler.scale(loss).backward()
                if self.clip_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(x)
                loss = self.loss_fn(preds, y)
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()

            # NaN guard: skip parameter update if loss exploded
            if torch.isnan(loss):
                self.optimizer.zero_grad()
                continue

            running_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                avg = running_loss / n_batches
                elapsed = time.time() - start
                print(
                    f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] avg_loss={avg:.6f} time={elapsed:.1f}s"
                )

        return {"train_loss": running_loss / max(1, n_batches)}

    def evaluate(
        self, val_loader: DataLoader, metric_fns: Optional[Dict[str, Callable]] = None
    ) -> Dict:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        metrics = {k: 0.0 for k in (metric_fns.keys() if metric_fns else [])}

        with torch.no_grad():
            for batch in val_loader:
                x, y = self._prepare_batch(batch)
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        preds = self.model(x)
                        loss = self.loss_fn(preds, y)
                else:
                    preds = self.model(x)
                    loss = self.loss_fn(preds, y)

                total_loss += loss.item()
                n_batches += 1

                if metric_fns:
                    for name, fn in metric_fns.items():
                        metrics[name] += fn(preds.detach().cpu(), y.detach().cpu())

        results = {"val_loss": total_loss / max(1, n_batches)}
        if metric_fns:
            results.update(
                {name: val / max(1, n_batches) for name, val in metrics.items()}
            )
        return results

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        validate_every: int = 1,
        checkpoint_every: int = 1,
        best_metric: Optional[str] = None,
        minimize_metric: bool = True,
    ):
        """
        Full train loop.

        best_metric: name returned by evaluate to track best model (e.g., "val_loss" or "val_mape").
        minimize_metric: if True lower is better.
        """
        best_val = float("inf") if minimize_metric else -float("inf")
        for epoch in range(1, epochs + 1):
            train_logs = self.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch} train_loss={train_logs['train_loss']:.6f}")

            if val_loader is not None and epoch % validate_every == 0:
                val_logs = self.evaluate(val_loader)
                print(f"Epoch {epoch} val_loss={val_logs['val_loss']:.6f}")
                metric_val = val_logs.get(best_metric, val_logs.get("val_loss"))
                if best_metric:
                    improved = (
                        (metric_val < best_val)
                        if minimize_metric
                        else (metric_val > best_val)
                    )
                    if improved:
                        best_val = metric_val
                        self.save_checkpoint(epoch, is_best=True)
                else:
                    # always save last
                    pass

            if epoch % checkpoint_every == 0:
                self.save_checkpoint(epoch, is_best=False)

        return

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(state, best_path)
        print(f"Saved checkpoint: {path}{' (best)' if is_best else ''}")

    def load_checkpoint(self, path: str, map_location: Optional[str] = None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch')})")
        return ckpt
