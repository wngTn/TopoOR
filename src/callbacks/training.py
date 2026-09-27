from __future__ import annotations

import logging
import math
import platform
import subprocess
import time
from collections import deque
from typing import Any

import cv2
import lightning.pytorch as L
import torch
from lightning.pytorch import Callback
from prettytable import PrettyTable
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn

from src.utils.experiment import now

log = logging.getLogger(__name__)


class EMACallback(L.Callback):
    """Exponential moving average of the model's parameters and buffers.

    Args:
        decay: EMA decay.
        warmup_steps: optimizer steps before averaging starts.
    """

    def __init__(self, decay: float, warmup_steps: int) -> None:
        super().__init__()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.ema_model: AveragedModel | None = None
        self._live_backup: dict[str, torch.Tensor] | None = None
        self._pending_ema_state: dict[str, Any] | None = None
        self._pending_ema_n: int | None = None
        self._last_step = -1

    def _is_warm(self, trainer: L.Trainer) -> bool:
        return trainer.global_step >= self.warmup_steps

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.ema_model is not None:
            return
        self.ema_model = AveragedModel(
            pl_module,
            avg_fn=get_ema_avg_fn(decay=self.decay),
            use_buffers=True,
        )
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        if self._pending_ema_state is not None:
            self.ema_model.module.load_state_dict(self._pending_ema_state)
            self.ema_model.n_averaged.fill_(self._pending_ema_n)  # type: ignore[arg-type]
            self._pending_ema_state = None
            self._pending_ema_n = None
        self._last_step = trainer.global_step

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.ema_model is None or not self._is_warm(trainer) or trainer.global_step == self._last_step:
            return
        self.ema_model.update_parameters(pl_module)
        self._last_step = trainer.global_step

    def on_train_start(self, trainer, pl_module):
        self._last_step = trainer.global_step

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.ema_model is None or not self._is_warm(trainer) or self._last_step <= 0:
            return
        self._live_backup = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}
        pl_module.load_state_dict(self.ema_model.module.state_dict(), strict=True)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._live_backup is None:
            return
        pl_module.load_state_dict(self._live_backup, strict=True)
        self._live_backup = None

    def on_save_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        if self.ema_model is None or not self._is_warm(trainer) or self._last_step <= 0:
            return
        # Evaluation loads EMA weights; resume restores the live weights.
        checkpoint["state_dict_live"] = checkpoint["state_dict"]
        checkpoint["state_dict"] = self.ema_model.module.state_dict()
        n = self.ema_model.n_averaged
        checkpoint["ema_n_averaged"] = int(n.item()) if torch.is_tensor(n) else int(n)

    def on_load_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        # BaseModule.on_load_checkpoint has already separated EMA and live weights.
        ema_state = checkpoint.get("state_dict_ema")
        if ema_state is None:
            return
        self._pending_ema_state = ema_state
        self._pending_ema_n = checkpoint["ema_n_averaged"]


def format_duration(seconds):
    if not math.isfinite(seconds):
        return "unknown"
    days, remainder = divmod(max(0, int(seconds)), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02}:{minutes:02}:{seconds:02}"
    return f"{days:02}:{clock}" if days else clock


class TrainingProgress(Callback):
    """Log the training losses, learning rate, throughput, peak VRAM and ETA.

    Losses are rolling means and throughput is measured over the last ``WINDOW`` optimizer steps.
    Unqualified ``loss_keys`` refer to ``train/<key>``.
    """

    WINDOW = 50

    def __init__(self, log_every_n_steps, loss_keys):
        self.log_every_n_steps = log_every_n_steps
        self.metrics = {name: deque(maxlen=self.WINDOW) for name in loss_keys}
        self.timings = deque(maxlen=self.WINDOW + 1)
        self.last_step = -1

    def on_train_start(self, trainer, pl_module):
        self.last_step = trainer.global_step
        self.timings.append((trainer.global_step, time.monotonic()))
        log.info("Training start | displayed metrics: rolling mean over %s optimizer steps", self.WINDOW)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == self.last_step:
            return
        self.last_step = step
        self.timings.append((step, time.monotonic()))
        for name, history in self.metrics.items():
            key = name if "/" in name else f"train/{name}"
            value = trainer.callback_metrics.get(key)
            if value is not None:
                history.append(torch.as_tensor(value).detach())
        if step % self.log_every_n_steps == 0:
            self._report(trainer, pl_module)

    def _report(self, trainer, module):
        step = trainer.global_step
        total = trainer.estimated_stepping_batches
        fields = [f"Step {step:,}/{int(total):,}" if math.isfinite(total) else f"Step {step:,}/unknown"]
        losses = [
            f"{name}={float(torch.stack(list(values)).mean().cpu()):.4f}"
            for name, values in self.metrics.items()
            if values
        ]
        if losses:
            fields.append(", ".join(losses))
        rates = [group["lr"] for optimizer in trainer.optimizers for group in optimizer.param_groups]
        fields.append(
            f"lr={rates[0]:.2e}" if len(rates) == 1 else ", ".join(f"lr/{i}={lr:.2e}" for i, lr in enumerate(rates))
        )
        first, last = self.timings[0], self.timings[-1]
        duration = (last[1] - first[1]) / max(1, last[0] - first[0])
        fields.append(f"{1 / max(duration, 1e-9):.2f} step/s")
        if module.device.type == "cuda":
            fields.append(f"VRAM peak {torch.cuda.max_memory_allocated(module.device) / 2**30:.1f} GiB")
        fields.append(f"ETA {format_duration((total - step) * duration)}")
        log.info(" | ".join(fields))

    def on_train_epoch_start(self, trainer, pl_module):
        log.info("Epoch %s start", trainer.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        log.info("Epoch %s completion", trainer.current_epoch)

    def on_train_end(self, trainer, pl_module):
        log.info("Training completion at step %s", trainer.global_step)

    def on_exception(self, trainer, pl_module, exception):
        log.error(
            "Keyboard interruption" if isinstance(exception, KeyboardInterrupt) else "Training failed",
            exc_info=(type(exception), exception, exception.__traceback__),
        )


def git_metadata():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        return f"{commit} (dirty={dirty})"
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


class StartupMetadata(L.Callback):
    def __init__(self, cfg):
        self.cfg = cfg

    def on_train_start(self, trainer, pl_module):
        cfg = self.cfg
        data = trainer.datamodule
        loader = trainer.train_dataloader
        batch_size = loader.batch_size
        parameters = sum(p.numel() for p in pl_module.parameters())
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        rows = {
            "Experiment": cfg.experiment_name,
            "Start time": now().isoformat(),
            "Seed": cfg.get("seed"),
            "Host": platform.node(),
            "Model": type(pl_module).__name__,
            "Dataset": type(loader.dataset).__name__,
            "Batch size": batch_size,
            "Loader workers / OpenCV threads": f"{loader.num_workers} / {cv2.getNumThreads()}",
            "Loader prefetch / pinned memory": f"{loader.prefetch_factor} / {loader.pin_memory}",
            "Effective batch size": batch_size * trainer.accumulate_grad_batches,
            "Epochs": trainer.max_epochs,
            "Expected optimizer steps": trainer.estimated_stepping_batches,
            "Optimizer": ", ".join(type(o).__name__ for o in trainer.optimizers),
            "Scheduler": ", ".join(type(s.scheduler).__name__ for s in trainer.lr_scheduler_configs) or "none",
            "Initial learning rates": [[g["lr"] for g in o.param_groups] for o in trainer.optimizers],
            "Precision": trainer.precision,
            "Accelerator": type(trainer.accelerator).__name__,
            "Gradient accumulation": trainer.accumulate_grad_batches,
            "Gradient-norm clipping": trainer.gradient_clip_val,
            "Resume checkpoint": trainer.ckpt_path or "none",
            "Parameters / trainable / non-trainable": f"{parameters:,} / {trainable:,} / {parameters - trainable:,}",
            "Python": platform.python_version(),
            "PyTorch": torch.__version__,
            "Lightning": L.__version__,
            "CUDA": torch.version.cuda or "unavailable",
            "Git": git_metadata(),
        }
        for split in ("train", "val"):
            rows[f"{split.capitalize()} samples"] = len(getattr(data, f"{split}_dataset"))
        if pl_module.device.type == "cuda":
            device = pl_module.device
            free, total = torch.cuda.mem_get_info(device)
            rows["GPU / free / total VRAM"] = (
                f"{torch.cuda.get_device_name(device)} / {free / 2**30:.1f} / {total / 2**30:.1f} GiB"
            )
        table = PrettyTable(["Setting", "Value"])
        table.align = "l"
        table.add_rows(list(rows.items()))
        log.info("Run metadata\n%s", table)
