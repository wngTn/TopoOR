"""Score Next Action and Robot Phase macro-F1; Robot Phase excludes 004_PKA."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import hydra
import lightning.pytorch as L
import rootutils
import torch
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch_geometric")
rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from src.callbacks.evaluation import write_metrics
from src.metrics import WorkflowRecognitionMetrics
from src.utils.experiment import configure_reproducibility, now, run_logging, unique_directory

log = logging.getLogger(__name__)


def _find_checkpoint(run_dir: Path, ckpt_name: str | None) -> Path:
    """Select a named checkpoint or the latest training step."""
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_name:
        return ckpt_dir / ckpt_name
    matches = sorted(ckpt_dir.glob("latest_step_*.ckpt"))
    if not matches:
        raise FileNotFoundError(f"no latest checkpoint in {ckpt_dir}")
    return matches[-1]


def _load_run_config(run_dir: Path):
    return OmegaConf.load(run_dir / "hparams.yaml")


def _evaluate_run(trainer, datamodule, run_dir, run_cfg, ckpt_name):
    ckpt_path = _find_checkpoint(run_dir, ckpt_name)
    weights = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = weights["global_step"]
    directory = unique_directory(run_dir / "test" / f"{step:06d}")
    with run_logging(run_dir):
        started = time.monotonic()
        log.info("Test start | run=%s | checkpoint=%s", run_dir.name, ckpt_path)
        model = hydra.utils.instantiate(run_cfg.model)
        metric, samples = WorkflowRecognitionMetrics(), 0
        for output in trainer.predict(model, datamodule=datamodule, ckpt_path=str(ckpt_path)):
            metric.update(**output)
            samples += len(output["sequences"])
        metrics = {name: float(value) for name, value in metric.compute().items()}
        metadata = dict(
            split="test",
            evaluation_timestamp=now().isoformat(),
            checkpoint_path=str(ckpt_path),
            global_step=step,
            epoch=weights["epoch"],
            num_samples=samples,
            duration_seconds=time.monotonic() - started,
        )
        write_metrics(directory, metadata, metrics)
        log.info("Test completion | %.2f seconds | %s", metadata["duration_seconds"], directory)
    return metrics, directory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score Next Action / Robot Phase macro-F1 of a workflow_recognition run"
    )
    parser.add_argument("--run", required=True, help="Run directory")
    parser.add_argument(
        "--ckpt-name", default=None, help="Checkpoint file name in <run>/checkpoints; default: the latest checkpoint"
    )
    parser.add_argument("--output", default=None, help="JSON output path (default: <test dir>/metrics.json)")
    parser.add_argument("--accelerator", default="auto", help="Lightning accelerator (auto/gpu/cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        sys.exit(f"Run dir not found: {run_dir}")
    run_cfg = _load_run_config(run_dir)
    configure_reproducibility(seed=run_cfg.get("seed"))
    datamodule = hydra.utils.instantiate(run_cfg.data)
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    metrics, output_dir = _evaluate_run(trainer, datamodule, run_dir, run_cfg, args.ckpt_name)
    log.info("%s: NA=%.4f  RP=%.4f", run_dir.name, metrics["next_action_macro_f1"], metrics["robot_phase_macro_f1"])
    out_path = Path(args.output) if args.output else output_dir / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"run": str(run_dir), **metrics}, f, indent=2)
    log.info("Saved metrics -> %s", out_path)


if __name__ == "__main__":
    main()
