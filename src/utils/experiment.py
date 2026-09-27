import logging
import os
import pickle
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import torch
from lightning.fabric.plugins.io import TorchCheckpointIO
from omegaconf import OmegaConf


def now():
    return datetime.now().astimezone()


def unique_directory(path):
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=False)
        return path
    except FileExistsError:
        child = path / now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        child.mkdir(parents=True, exist_ok=False)
        return child


def experiment_directory(root, name, policy):
    if policy not in {"error", "resume", "new_run"}:
        raise ValueError(f"Unknown existing_experiment_policy: {policy}")
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("experiment_name must be a single directory name")
    path = Path(root).expanduser().resolve() / name
    if policy == "new_run":
        path = unique_directory(path)
    elif policy == "error":
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"Experiment already exists: {path}. Set logging.existing_experiment_policy=resume or new_run."
            ) from error
    else:
        path.mkdir(parents=True, exist_ok=True)
    return str(path)


def hydra_metadata_directory(output_dir):
    if (Path(output_dir) / ".hydra" / "config.yaml").exists():
        return f".hydra/{now().strftime('%Y-%m-%d_%H-%M-%S_%f')}"
    return ".hydra"


def register_resolvers():
    OmegaConf.register_new_resolver("experiment_dir", experiment_directory, replace=True, use_cache=True)
    OmegaConf.register_new_resolver("experiment_metadata", hydra_metadata_directory, replace=True, use_cache=True)


def save_config(cfg, directory):
    directory = Path(directory)
    for name in ("checkpoints", "wandb", "val"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    path = directory / "hparams.yaml"
    if path.exists():
        archive = directory / ".hydra" / f"hparams_{now().strftime('%Y-%m-%d_%H-%M-%S_%f')}.yaml"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("xb") as stream, path.open("rb") as previous:
            shutil.copyfileobj(previous, stream)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, prefix=".hparams_", delete=False) as stream:
        OmegaConf.save(cfg, stream, resolve=True)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def configure_reproducibility(seed=None):
    """Select deterministic kernels and seed every RNG."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    if seed is not None:
        import lightning.pytorch as L

        L.seed_everything(int(seed), workers=True)


class RunFormatter(logging.Formatter):
    """Prefix every line of a record, continuation lines included, with its local time."""

    def __init__(self):
        super().__init__("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    def format(self, record):
        return super().format(record).replace("\n", f"\n[{self.formatTime(record, self.datefmt)}] ")


@contextmanager
def run_logging(directory):
    """Send INFO logging to the console and to a new ``train_<time>.log`` in ``directory``."""
    root = logging.getLogger()
    previous = root.handlers[:], root.level, sys.excepthook
    moment = now()
    path = directory / f"train_{moment:%Y-%m-%d_%H-%M-%S}.log"
    while path.exists():
        moment += timedelta(seconds=1)
        path = directory / f"train_{moment:%Y-%m-%d_%H-%M-%S}.log"
    handlers = [logging.StreamHandler(), logging.FileHandler(path, mode="x", encoding="utf-8")]
    for handler in handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(RunFormatter())
    root.handlers = handlers
    root.setLevel(logging.INFO)
    lightning_loggers = [logging.getLogger(name) for name in ("lightning", "lightning.pytorch", "lightning.fabric")]
    saved = [(logger, logger.handlers[:], logger.propagate) for logger in lightning_loggers]
    for logger in lightning_loggers:
        logger.handlers = []
        logger.propagate = True
    logging.captureWarnings(True)
    sys.excepthook = lambda *exc: root.critical("Unhandled exception", exc_info=exc)
    try:
        yield
    except KeyboardInterrupt:
        root.warning("Keyboard interruption")
        raise
    except BaseException:
        root.exception("Unhandled exception")
        raise
    finally:
        logging.captureWarnings(False)
        for handler in handlers:
            handler.flush()
            handler.close()
        root.handlers, root.level, sys.excepthook = previous
        for logger, old_handlers, propagate in saved:
            logger.handlers, logger.propagate = old_handlers, propagate


def newest_checkpoint(paths):
    valid = []
    for path in paths:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if "state_dict" not in checkpoint:
                raise ValueError("Missing model state")
            valid.append((int(checkpoint["global_step"]), Path(path)))
        except (OSError, ValueError, KeyError, RuntimeError, EOFError, TypeError, pickle.UnpicklingError) as error:
            logging.getLogger(__name__).warning("Ignoring invalid resume checkpoint %s: %s", path, error)
    if not valid:
        raise FileNotFoundError("No valid resume checkpoint; provide ckpt_path explicitly")
    return max(valid)[1]


class AtomicCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        if storage_options is not None:
            raise TypeError("Local checkpoints do not accept storage_options")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".checkpoint_", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
