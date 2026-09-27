"""Hydra and Lightning training entry point."""

import logging
import time
from pathlib import Path

import hydra
import lightning.pytorch as L
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.callbacks.training import StartupMetadata
from src.utils.experiment import (
    configure_reproducibility,
    newest_checkpoint,
    register_resolvers,
    run_logging,
    save_config,
)

log = logging.getLogger(__name__)
register_resolvers()


def _instantiate_components(cfg):
    model = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.data)
    logger = hydra.utils.instantiate(cfg.logger) if cfg.get("logger") else False
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    trainer.callbacks.insert(0, StartupMetadata(cfg))
    return model, datamodule, trainer


def run(cfg):
    configure_reproducibility(seed=cfg.seed)
    model, datamodule, trainer = _instantiate_components(cfg)
    checkpoint = cfg.get("ckpt_path") or None
    if cfg.logging.existing_experiment_policy == "resume" and checkpoint is None:
        candidates = sorted((Path(cfg.paths.output_dir) / "checkpoints").glob("latest_step_*.ckpt"))
        if not candidates:
            raise FileNotFoundError("Resume requires ckpt_path or an existing latest checkpoint")
        checkpoint = str(newest_checkpoint(candidates))
    if checkpoint:
        log.info("Resume from checkpoint: %s", checkpoint)
    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)
    finally:
        for logger in trainer.loggers:
            logger.finalize("failed" if trainer.interrupted else "success")
            if isinstance(logger, L.loggers.WandbLogger):
                logger.experiment.finish()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    directory = Path(cfg.paths.output_dir)
    started = time.monotonic()
    with run_logging(directory):
        try:
            save_config(cfg, directory)
            run(cfg)
        finally:
            log.info("Total elapsed runtime: %.2f seconds", time.monotonic() - started)


if __name__ == "__main__":
    main()
