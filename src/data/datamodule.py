from __future__ import annotations

from functools import cached_property

import lightning.pytorch as L
import numpy as np
from torch.utils.data import DataLoader, Sampler
from torch_geometric.data import Batch, Data

from src.data.temporal_graph import TemporalSuperGraph


class DataModule(L.LightningDataModule):
    """Train and validation MM-OR windows; prediction uses the validation set."""

    def __init__(self, train_cfg, val_cfg):
        super().__init__()
        self.train_cfg, self.val_cfg = train_cfg, val_cfg

    @cached_property
    def train_dataset(self):
        return self.train_cfg.dataset(split="train")

    @cached_property
    def val_dataset(self):
        return self.val_cfg.dataset(split="test")

    def setup(self, stage=None):
        if stage in (None, "fit"):
            _ = self.train_dataset
        _ = self.val_dataset

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, Data):
            return batch.to(device, non_blocking=device.type == "cuda")
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _loader(self, dataset, cfg):
        loader_params = dict(cfg.loader_params)
        sampler = getattr(cfg, "sampler", None)
        if sampler is not None:
            loader_params["sampler"] = sampler(dataset)
        if loader_params.get("num_workers", 0) == 0:
            loader_params.update(prefetch_factor=None, persistent_workers=False)
        return DataLoader(dataset, collate_fn=collate_fn, **loader_params)

    def train_dataloader(self):
        return self._loader(self.train_dataset, self.train_cfg)

    def val_dataloader(self):
        return self._loader(self.val_dataset, self.val_cfg)

    predict_dataloader = val_dataloader


def collate_fn(batch: list[TemporalSuperGraph]) -> Batch:
    return Batch.from_data_list(batch)


class EpochSeededSampler(Sampler):
    """Shuffles each epoch with numpy ``SeedSequence([seed, epoch])``; draws no global RNG."""

    def __init__(self, dataset, seed=0):
        self.size = len(dataset)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        indices = np.arange(self.size)
        generator.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return self.size
