"""Score relation macro-F1 on the held-out takes and save per-pair logits and labels."""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import rootutils
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch_geometric")

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.data.datasets.constants import NUM_PREDICATES
from src.eval import _find_checkpoint, _load_run_config  # noqa: E402
from src.utils.experiment import configure_reproducibility

log = logging.getLogger(__name__)


def _uncalibrated_macro_f1(logits: np.ndarray, labels: np.ndarray):
    """Match RelationPredictionMetrics: average supported classes, including none (16)."""
    preds = logits.argmax(-1)
    per_class_f1 = np.zeros(NUM_PREDICATES, dtype=np.float64)
    has_support = np.zeros(NUM_PREDICATES, dtype=bool)
    for c in range(NUM_PREDICATES):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if (precision + recall) > 1e-8:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        per_class_f1[c] = f1
        has_support[c] = (tp + fn) > 0
    if has_support.any():
        macro = float(per_class_f1[has_support].mean())
    else:
        macro = 0.0
    return macro, per_class_f1, has_support


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a relation_prediction run and dump its logits to an .npz")
    parser.add_argument("--run", required=True, help="relation_prediction run directory")
    parser.add_argument(
        "--ckpt-name", default=None,
        help="Checkpoint file name in <run>/checkpoints; default: the latest checkpoint",
    )
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--accelerator", default="auto", help="Lightning accelerator (auto/gpu/cpu)")
    args = parser.parse_args()

    configure_reproducibility()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    run = Path(args.run)
    if not run.is_dir():
        raise SystemExit(f"Run dir not found: {run}")

    cfg = _load_run_config(run)
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    ckpt = _find_checkpoint(Path(run), args.ckpt_name)
    log.info("run:  %s", run.name)
    log.info("ckpt: %s", ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
    )

    outs = trainer.predict(model, datamodule=datamodule)
    if not outs:
        raise SystemExit("trainer.predict returned no batches")

    # Logits and labels align per pair; sequences are per sample.
    logit_batches = [b["relation_logits"] for b in outs if b["relation_logits"].numel() > 0]
    label_batches = [b["relation_labels"] for b in outs if b["relation_labels"].numel() > 0]
    if not logit_batches or not label_batches:
        raise SystemExit("no relation pairs found across all predict batches")

    logits = torch.cat(logit_batches, dim=0).cpu().numpy().astype(np.float32)  # [N, 17]
    labels = torch.cat(label_batches, dim=0).cpu().numpy().astype(np.int64)    # [N]

    if logits.shape[0] != labels.shape[0]:
        raise SystemExit(
            f"logits/labels pair-count mismatch: {logits.shape[0]} vs {labels.shape[0]}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, logits=logits, labels=labels)
    log.info("saved -> %s", out_path)

    support = np.bincount(labels, minlength=NUM_PREDICATES)
    macro, per_class_f1, has_support = _uncalibrated_macro_f1(logits, labels)

    log.info("logits.shape = %s", logits.shape)
    log.info("labels.shape = %s", labels.shape)
    log.info("per-class support (np.bincount, minlength=17): %s", support.tolist())
    log.info("UNCALIBRATED argmax macro-F1 (incl none id 16): %.6f", macro)
    log.info("per-class F1 (argmax):")
    log.info("  %-5s %-10s %-10s", "class", "support", "f1")
    for c in range(NUM_PREDICATES):
        flag = "" if has_support[c] else "  (no support, excluded)"
        log.info("  %-5d %-10d %-10.6f%s", c, int(support[c]), per_class_f1[c], flag)


if __name__ == "__main__":
    main()
