"""Task-agnostic supervised Trainer for StochAct or regular PyTorch networks.

The Trainer is optional. Users may train StochasticModel with their existing loops.
Detailed paper metrics, OOD and calibration belong to the future Evaluator phase.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from ..core.activations import StochasticActivation
from ..uncertainty.inference import _extract_output, _inference_mode
from ..uncertainty import predict_mc
from .history import TrainingHistory


def _prepare_inputs(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _prepare_inputs(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_prepare_inputs(item, device) for item in value)
    return value


def _batch_parts(batch: Any) -> tuple[Any, torch.Tensor]:
    if isinstance(batch, Mapping):
        input_key = next((key for key in ("inputs", "input", "x", "image", "images") if key in batch), None)
        target_key = next((key for key in ("targets", "target", "y", "label", "labels", "mask", "masks") if key in batch), None)
        if input_key is None or target_key is None:
            raise ValueError("Mapping batches must contain an input key and a target key")
        x, y = batch[input_key], batch[target_key]
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        x, y = batch
    else:
        raise ValueError("Each batch must be (inputs, targets) or a recognized mapping")
    if not torch.is_tensor(y):
        raise TypeError("Training targets must be a Tensor")
    return x, y


def _forward(model: nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, (list, tuple)):
        return model(*inputs)
    return model(inputs)


def _binary_output(output: torch.Tensor, task: str) -> bool:
    if task == "classification":
        if output.ndim == 1:
            return True
        if output.ndim != 2:
            raise ValueError("Classification outputs must be [B], [B,1], or [B,C]")
        return output.shape[1] == 1
    if task == "segmentation":
        if output.ndim < 3:
            raise ValueError("Segmentation outputs must be [B,*spatial] or [B,C,*spatial]")
        return output.ndim == 3 or output.shape[1] == 1
    return False


def _binary_targets(targets: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    if targets.shape == output.shape:
        return targets.float()
    if output.ndim >= 2 and output.shape[1] == 1 and targets.shape == (output.shape[0], *output.shape[2:]):
        return targets.unsqueeze(1).float()
    raise ValueError(f"Binary output shape {tuple(output.shape)} incompatible with targets {tuple(targets.shape)}")


def _class_targets(targets: torch.Tensor, output: torch.Tensor, task: str) -> torch.Tensor:
    wanted_shape = (output.shape[0], *output.shape[2:])
    if targets.shape != wanted_shape:
        raise ValueError(f"{task} class targets must be {wanted_shape}, got {tuple(targets.shape)}")
    return targets.long()


class _TaskStats:
    """Batch-size-aware metrics; no growing tensors or sklearn dependency."""

    def __init__(self, task: str, *, from_logits: bool, threshold: float, ignore_index: int | None):
        self.task = task
        self.from_logits = from_logits
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.total = 0
        self.correct = 0
        self.absolute_error = 0.0
        self.sq_error = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    @torch.no_grad()
    def update(self, output: torch.Tensor, targets: torch.Tensor) -> None:
        if self.task == "regression":
            if output.shape != targets.shape:
                raise ValueError("Regression outputs and targets must have identical shapes")
            delta = (output.float() - targets.float())
            self.absolute_error += delta.abs().sum().item()
            self.sq_error += delta.square().sum().item()
            self.total += delta.numel()
            return
        binary = _binary_output(output, self.task)
        if binary:
            target = _binary_targets(targets, output)
            prob = output.sigmoid() if self.from_logits else output
            pred = (prob >= self.threshold).long()
            if not torch.all((target == 0) | (target == 1)):
                if self.ignore_index is None or not torch.all((target == 0) | (target == 1) | (target == self.ignore_index)):
                    raise ValueError("Binary targets must be 0/1, optionally including ignore_index")
        else:
            target = _class_targets(targets, output, self.task)
            pred = output.argmax(dim=1)
        target = target.long()
        valid = torch.ones_like(target, dtype=torch.bool) if self.ignore_index is None else target != self.ignore_index
        self.correct += ((pred == target) & valid).sum().item()
        self.total += valid.sum().item()
        if self.task == "segmentation":
            # Class-sensitive, micro foreground Dice (exclude background class 0).
            # Incorrect foreground-to-foreground predictions count as both FP and FN.
            mismatch = (pred != target) & valid
            self.tp += ((pred == target) & (target != 0) & valid).sum().item()
            self.fp += (mismatch & (pred != 0)).sum().item()
            self.fn += (mismatch & (target != 0)).sum().item()

    def compute(self) -> dict[str, float]:
        if self.total == 0:
            raise ValueError("No valid targets were evaluated")
        if self.task == "regression":
            return {"mae": self.absolute_error / self.total, "mse": self.sq_error / self.total}
        metrics = {"accuracy" if self.task == "classification" else "pixel_accuracy": self.correct / self.total}
        if self.task == "segmentation":
            denom = 2 * self.tp + self.fp + self.fn
            metrics["foreground_dice"] = 2 * self.tp / denom if denom else 1.0
        return metrics


class Trainer:
    """General supervised training utility with optional MC uncertainty diagnostics.

    Supports classification, segmentation, and regression; no separate task wrappers.
    Logs train/validation loss and lightweight metrics. For calibration, OOD,
    and detailed task metrics, use the separate Evaluator.

    A custom loss_fn takes (model_output, targets) and must return a scalar.
    Optimizers/schedulers may be supplied as PyTorch instances or standard names.
    Default losses expect raw logits for classification/segmentation, MSE for regression.
    When `from_logits=False`, probability-safe losses are used instead.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        task: str = "classification",
        optimizer: str | Optimizer = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler: str | LRScheduler | ReduceLROnPlateau | None = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | str | None = None,
        mixed_precision: bool = False,
        grad_clip_norm: float | None = None,
        accumulation_steps: int = 1,
        from_logits: bool = True,
        threshold: float = 0.5,
        ignore_index: int | None = None,
        output_key: str | None = None,
        output_index: int | None = None,
        seed: int | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module")
        if task not in ("classification", "segmentation", "regression"):
            raise ValueError("task must be classification, segmentation or regression")
        if not isinstance(accumulation_steps, int) or isinstance(accumulation_steps, bool) or accumulation_steps < 1:
            raise ValueError("accumulation_steps must be a positive integer")
        if lr <= 0 or not math.isfinite(lr):
            raise ValueError("lr must be finite and positive")
        if weight_decay < 0 or not math.isfinite(weight_decay):
            raise ValueError("weight_decay must be finite and nonnegative")
        if grad_clip_norm is not None and (grad_clip_norm <= 0 or not math.isfinite(grad_clip_norm)):
            raise ValueError("grad_clip_norm must be positive")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0,1]")
        if output_key is not None and output_index is not None:
            raise ValueError("Specify output_key or output_index, not both")
        if loss_fn is not None and not callable(loss_fn):
            raise TypeError("loss_fn must be callable")
        if seed is not None:
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError("seed must be a nonnegative integer")
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.model = model
        self.task = task
        self.device = torch.device(device) if device is not None else next(model.parameters(), torch.empty(0)).device
        self.model.to(self.device)
        self.from_logits = from_logits
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.output_key = output_key
        self.output_index = output_index
        self.loss_fn = loss_fn
        self.mixed_precision = mixed_precision
        self.grad_clip_norm = grad_clip_norm
        self.accumulation_steps = accumulation_steps
        self._scheduler_spec = scheduler
        self.scheduler: LRScheduler | ReduceLROnPlateau | Any | None = scheduler if not isinstance(scheduler, str) else None
        if isinstance(optimizer, str):
            choices = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
            if optimizer.lower() not in choices:
                raise ValueError("optimizer must be adam, adamw, sgd, or Optimizer")
            self.optimizer = choices[optimizer.lower()](model.parameters(), lr=lr, weight_decay=weight_decay)
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise TypeError("optimizer must be a name or torch.optim.Optimizer")
        if self.device.type not in ("cuda", "cpu") and mixed_precision:
            raise ValueError("Mixed precision is currently supported on CPU/CUDA only")
        # CPU autocast uses bfloat16; CUDA autocast uses float16 with gradient scaling.
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and self.device.type == "cuda")
        except (TypeError, AttributeError):  # compatibility with supported earlier PyTorch
            self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision and self.device.type == "cuda")
        self.history = TrainingHistory()
        self.current_epoch = 0
        self._best_bad_epochs = 0

    def _autocast(self):
        if not self.mixed_precision:
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
        )

    def _loss(self, output: torch.Tensor, targets: torch.Tensor, *, from_logits: bool | None = None) -> torch.Tensor:
        if from_logits is None:
            from_logits = self.from_logits
        if self.loss_fn is not None:
            result = self.loss_fn(output, targets)
            if not torch.is_tensor(result) or result.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")
            return result
        if self.task == "regression":
            if output.shape != targets.shape:
                raise ValueError("Regression output and target shapes must match")
            return nn.functional.mse_loss(output.float(), targets.float())
        binary = _binary_output(output, self.task)
        if binary:
            target = _binary_targets(targets, output)
            if self.ignore_index is not None:
                valid = target != self.ignore_index
                if not bool(valid.any()):
                    raise ValueError("Binary batch has no valid targets")
                target = torch.where(valid, target, torch.zeros_like(target))
            else:
                valid = None
            if not torch.all((target == 0) | (target == 1)):
                raise ValueError("Binary targets must be 0/1")
            losses = (nn.functional.binary_cross_entropy_with_logits(output.float(), target, reduction="none")
                      if from_logits else nn.functional.binary_cross_entropy(output.float(), target, reduction="none"))
            return losses[valid].mean() if valid is not None else losses.mean()
        target = _class_targets(targets, output, self.task)
        ignore = self.ignore_index if self.ignore_index is not None else -100
        if self.ignore_index is not None and not bool((target != self.ignore_index).any()):
            raise ValueError("Multiclass batch has no valid targets")
        if from_logits:
            return nn.functional.cross_entropy(output.float(), target, ignore_index=ignore)
        # Probabilities-only models should not pass through cross_entropy as logits.
        log_probs = output.float().clamp_min(1e-8).log()
        return nn.functional.nll_loss(log_probs, target, ignore_index=ignore)

    def _step_optimizer(self, pending: int) -> None:
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        # Losses were divided by accumulation_steps. Adjust tail group when
        # number of batches is not divisible by accumulation_steps.
        if pending != self.accumulation_steps:
            ratio = self.accumulation_steps / pending
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(ratio)
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _epoch(self, loader: Any, *, training: bool) -> dict[str, float]:
        if training:
            self.model.train()
            # Stochastic modules explicitly ON even if user had disabled them.
            stochastic = [m for m in self.model.modules() if isinstance(m, StochasticActivation)]
            for m in stochastic:
                m.set_stochastic(True)
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()
            stochastic = [m for m in self.model.modules() if isinstance(m, StochasticActivation)]
            for m in stochastic:
                m.set_stochastic(True)
        stats = _TaskStats(self.task, from_logits=self.from_logits, threshold=self.threshold,
                           ignore_index=self.ignore_index)
        total_loss = 0.0
        total_examples = 0
        pending = 0
        grad_ctx = torch.enable_grad() if training else torch.no_grad()
        with grad_ctx:
            for batch in loader:
                x, target = _batch_parts(batch)
                x = _prepare_inputs(x, self.device)
                target = target.to(self.device, non_blocking=True)
                with self._autocast():
                    output = _extract_output(_forward(self.model, x), self.output_key, self.output_index)
                    loss = self._loss(output, target)
                if not bool(torch.isfinite(loss).all()):
                    raise ValueError("Nonfinite loss encountered")
                size = target.shape[0]
                total_examples += size
                total_loss += float(loss.detach()) * size
                stats.update(output.detach(), target.detach())
                if training:
                    self.scaler.scale(loss / self.accumulation_steps).backward()
                    pending += 1
                    if pending == self.accumulation_steps:
                        self._step_optimizer(pending)
                        pending = 0
            if training and pending:
                self._step_optimizer(pending)
        if total_examples == 0:
            raise ValueError("Loader returned no batches")
        return {"loss": total_loss / total_examples, **stats.compute()}

    def _make_scheduler(self, epochs: int) -> None:
        if self.scheduler is not None:
            return
        choice = self._scheduler_spec
        if choice is None:
            return
        if isinstance(choice, str):
            key = choice.lower()
            if key == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
            elif key == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(1, epochs // 3), gamma=0.1)
            elif key == "plateau":
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=2)
            else:
                raise ValueError("scheduler must be None, cosine, step, plateau, or scheduler object")
        elif not hasattr(choice, "step") or not hasattr(choice, "state_dict"):
            raise TypeError("scheduler must support step() and state_dict()")

    def _uncertainty_score(self, loader: Any, *, n_samples: int, max_batches: int) -> float:
        """Mean per-example predictive variance, not a calibration or error metric."""
        if not any(isinstance(m, StochasticActivation) for m in self.model.modules()):
            raise ValueError("Uncertainty monitoring requires StochasticActivation layers")
        total = 0.0
        examples = 0
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            inputs, _ = _batch_parts(batch)
            # predict_mc supports a single Tensor input. Structured inputs still
            # work in ordinary training/validation but not MC diagnostics.
            if not torch.is_tensor(inputs):
                raise TypeError("Uncertainty diagnostics require tensor inputs")
            inputs = inputs.to(self.device, non_blocking=True)
            result = predict_mc(self.model, inputs, task=self.task, n_samples=n_samples,
                                from_logits=self.from_logits, threshold=self.threshold,
                                output_key=self.output_key, output_index=self.output_index)
            total += float(result.uncertainty.sum())
            examples += result.uncertainty.numel()
        if not examples:
            raise ValueError("Uncertainty monitor loader returned no batches")
        return total / examples

    def fit(
        self,
        train_loader: Any,
        val_loader: Any | None = None,
        *,
        epochs: int,
        uncertainty_every: int | None = None,
        uncertainty_samples: int = 5,
        uncertainty_batches: int = 1,
        checkpoint_dir: str | Path | None = None,
        early_stopping_patience: int | None = None,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True,
        callbacks: Sequence[Callable[["Trainer", dict[str, float]], None]] | None = None,
    ) -> TrainingHistory:
        """Fit for `epochs` additional epochs, updating `history` in place.

        Computes uncertainty on the *first* `uncertainty_batches` of each split
        every `uncertainty_every` epochs (also at epoch 1). Validation and
        diagnostics do not update BatchNorm statistics or gradients.
        Each callback receives (trainer, epoch_metrics_copy) after checkpointing.

        Best checkpoint is selected on validation loss (or training loss when
        no validation set), not on uncertainty score. `last.pt` always captures
        the actual final epoch, even if best weights are restored afterwards.
        """
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if uncertainty_every is not None and (not isinstance(uncertainty_every, int) or isinstance(uncertainty_every, bool) or uncertainty_every < 1):
            raise ValueError("uncertainty_every must be a positive integer or None")
        if uncertainty_samples < 1 or uncertainty_batches < 1:
            raise ValueError("uncertainty_samples and uncertainty_batches must be positive")
        if early_stopping_patience is not None and (not isinstance(early_stopping_patience, int) or early_stopping_patience < 1):
            raise ValueError("early_stopping_patience must be a positive integer or None")
        if min_delta < 0:
            raise ValueError("min_delta must be nonnegative")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be bool")
        if callbacks is not None and (not isinstance(callbacks, (list, tuple)) or not all(callable(cb) for cb in callbacks)):
            raise TypeError("callbacks must be a list/tuple of callables")
        path = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
        self._make_scheduler(self.current_epoch + epochs)
        best_state = None  # Updated only when this fit call achieves a new best.
        final_epoch = self.current_epoch + epochs
        for _ in range(epochs):
            epoch = self.current_epoch + 1
            train_log = self._epoch(train_loader, training=True)
            val_log = self._epoch(val_loader, training=False) if val_loader is not None else None
            record = {"epoch": float(epoch), **{f"train_{k}": v for k, v in train_log.items()}}
            if val_log is not None:
                record.update({f"val_{k}": v for k, v in val_log.items()})
            if uncertainty_every is not None and (epoch == 1 or epoch % uncertainty_every == 0):
                record["train_uncertainty"] = self._uncertainty_score(
                    train_loader, n_samples=uncertainty_samples, max_batches=uncertainty_batches)
                if val_loader is not None:
                    record["val_uncertainty"] = self._uncertainty_score(
                        val_loader, n_samples=uncertainty_samples, max_batches=uncertainty_batches)
            monitored = record["val_loss"] if val_log is not None else record["train_loss"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(monitored)
                else:
                    self.scheduler.step()
            record["lr"] = float(self.optimizer.param_groups[0]["lr"])
            if self.history.best_value is None or monitored < self.history.best_value - min_delta:
                self.history.best_value = monitored
                self.history.best_epoch = epoch
                self._best_bad_epochs = 0
                if restore_best_weights:
                    import copy
                    best_state = copy.deepcopy(self.model.state_dict())
                improved = True
            else:
                self._best_bad_epochs += 1
                improved = False
            self.history.records.append(record)
            self.current_epoch = epoch
            if path is not None:
                self.save_checkpoint(path / "last.pt")
                if improved:
                    self.save_checkpoint(path / "best.pt")
            if callbacks is not None:
                for callback in callbacks:
                    callback(self, dict(record))
            if verbose:
                print("Epoch {}/{} | ".format(epoch, final_epoch)
                      + " | ".join(f"{k}={v:.4f}" for k, v in record.items() if k != "epoch"))
            if early_stopping_patience is not None and self._best_bad_epochs >= early_stopping_patience:
                self.history.stopped_early = True
                break
        if restore_best_weights and best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    def validate(self, loader: Any) -> dict[str, float]:
        """Evaluate a loader with one stochastic forward per batch."""
        return self._epoch(loader, training=False)

    def test(self, loader: Any, *, n_samples: int = 1) -> dict[str, float]:
        """Evaluate test loss/metrics; n_samples>1 uses MC mean predictions.

        For more detailed test metrics, use the separate Evaluator.
        """
        if not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be positive")
        if n_samples == 1:
            return self.validate(loader)
        stats = _TaskStats(self.task, from_logits=False if self.task != "regression" else self.from_logits,
                           threshold=self.threshold, ignore_index=self.ignore_index)
        total_loss = 0.0
        total_examples = 0
        with torch.no_grad():
            for batch in loader:
                x, target = _batch_parts(batch)
                if not torch.is_tensor(x):
                    raise TypeError("MC test requires Tensor inputs")
                x = x.to(self.device)
                target = target.to(self.device)
                result = predict_mc(self.model, x, task=self.task, n_samples=n_samples,
                                    from_logits=self.from_logits, threshold=self.threshold,
                                    output_key=self.output_key, output_index=self.output_index)
                # Loss on predictive-mean probabilities (not mean per-pass loss).
                # Custom loss_fn is interpreted as a function of these mean predictions.
                if self.loss_fn is not None:
                    loss = self.loss_fn(result.mean, target)
                else:
                    loss = self._loss(result.mean, target, from_logits=False)
                if not bool(torch.isfinite(loss).all()):
                    raise ValueError("Nonfinite MC test loss")
                total_loss += float(loss) * target.shape[0]
                total_examples += target.shape[0]
                stats.update(result.mean, target)
        if not total_examples:
            raise ValueError("Test loader returned no batches")
        return {"loss": total_loss / total_examples, **stats.compute()}

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save model, optimizer, scheduler, AMP scaler, RNG and epoch history."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "epoch": self.current_epoch,
            "history": self.history.as_dict(),
            "bad_epochs": self._best_bad_epochs,
            "task": self.task,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng": random.getstate(),
        }
        torch.save(payload, out)
        return out

    def load_checkpoint(self, path: str | Path, *, load_optimizer: bool = True, restore_rng: bool = False) -> None:
        """Restore a *trusted* StochAct checkpoint (torch.load uses pickle).

        Initialize Trainer with compatible optimizer and scheduler configuration
        before loading. If the checkpoint has scheduler state and this Trainer's
        scheduler is not constructed yet, provide the same scheduler name and it
        will be constructed using the saved epoch count as a provisional horizon.
        The saved cosine scheduler state includes its original T_max horizon.
        Do not load untrusted checkpoint files: torch.load uses pickle.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("task") != self.task:
            raise ValueError("Checkpoint task differs from Trainer task")
        if self.scheduler is None and checkpoint.get("scheduler") is not None:
            self._make_scheduler(max(1, int(checkpoint["epoch"])))
        self.model.load_state_dict(checkpoint["model"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler is not None and checkpoint.get("scheduler") is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        self.current_epoch = int(checkpoint["epoch"])
        self.history = TrainingHistory.from_dict(checkpoint["history"])
        self._best_bad_epochs = int(checkpoint.get("bad_epochs", 0))
        if restore_rng:
            torch.set_rng_state(checkpoint["torch_rng"].cpu())
            if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
            random.setstate(checkpoint["python_rng"])
