import logging
import time
from pathlib import Path

import torch
from lightning.pytorch import Callback
from lightning.pytorch.utilities.data import extract_batch_size
from omegaconf import OmegaConf
from prettytable import PrettyTable

from src.utils.experiment import now, unique_directory

log = logging.getLogger(__name__)


def write_metrics(directory, metadata, metrics):
    details = PrettyTable(["Setting", "Value"])
    details.align = "l"
    details.add_rows(list(metadata.items()))
    table = PrettyTable(["Metric", "Value"])
    table.align = "l"
    table.add_rows([[name, f"{value:.4f}"] for name, value in sorted(metrics.items())])
    with (directory / "metrics.log").open("x", encoding="utf-8") as stream:
        stream.write(f"{details}\n{table}\n")
    with (directory / "metrics.yaml").open("x", encoding="utf-8") as stream:
        OmegaConf.save(OmegaConf.create({**metadata, "metrics": metrics}), stream)
    log.info("%s metrics\n%s\n%s", metadata["split"], details, table)


class EvaluationArtifacts(Callback):
    """Write each validation pass to ``<directory>/val/<step>/metrics.{log,yaml}``."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def on_validation_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self.started = time.monotonic()
        self.samples = 0
        for key in list(trainer.callback_metrics):
            if key.startswith("val/"):
                trainer.callback_metrics.pop(key, None)
        step = trainer.global_step
        self.output_dir = unique_directory(self.directory / "val" / f"{step:06d}")
        log.info("Val start | step=%s", step)
        self.metadata = dict(
            split="val", evaluation_timestamp=now().isoformat(), global_step=step, epoch=trainer.current_epoch
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        self.samples += int(batch.num_graphs) if hasattr(batch, "num_graphs") else extract_batch_size(batch)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = {
            key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            for key, value in pl_module.evaluation_metrics.items()
            if key.startswith("val/")
        }
        self.metadata.update(num_samples=self.samples, duration_seconds=time.monotonic() - self.started)
        write_metrics(self.output_dir, self.metadata, metrics)
        log.info(
            "Val completion | %s samples | %.2f seconds | %s",
            self.samples,
            self.metadata["duration_seconds"],
            self.output_dir,
        )
