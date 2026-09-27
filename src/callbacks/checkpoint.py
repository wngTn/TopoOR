"""Latest-step checkpoint retention using Lightning's full-state checkpoints."""

import logging
from pathlib import Path

from lightning.pytorch.callbacks import Checkpoint

log = logging.getLogger(__name__)


class ExperimentCheckpoint(Checkpoint):
    """Keep one checkpoint, ``latest_step_<step>.ckpt``, of the newest optimizer step.

    It is written every ``every_n_train_steps`` optimizer steps and at the end of training,
    and rewritten after a validation at the same step so that it carries the finished
    validation loop state.
    """

    def __init__(self, dirpath, every_n_train_steps):
        self.dirpath = Path(dirpath)
        self.every_n_train_steps = every_n_train_steps
        self.last_model_path = ""
        self._latest_step = 0

    def on_train_start(self, trainer, pl_module):
        self._latest_step = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step > self._latest_step and step % self.every_n_train_steps == 0:
            self._save(trainer)

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking and trainer.global_step == self._latest_step > 0:
            self._save(trainer)

    def on_train_end(self, trainer, pl_module):
        if trainer.global_step > self._latest_step:
            self._save(trainer)

    def _save(self, trainer):
        path = self.dirpath / f"latest_step_{trainer.global_step:06d}.ckpt"
        trainer.save_checkpoint(path)
        for stale in self.dirpath.glob("latest_step_*.ckpt"):
            if stale != path:
                stale.unlink()
        self.last_model_path, self._latest_step = str(path), trainer.global_step
        log.info("Latest checkpoint %s | step=%s epoch=%s", path, trainer.global_step, trainer.current_epoch)
