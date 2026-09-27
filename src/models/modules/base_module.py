from typing import Any

import lightning.pytorch as L
import pytorch_warmup
import torch
from torch import nn


class BaseModule(L.LightningModule):
    def __init__(self, optimizer, scheduler, metric, losses, warmup_steps=0):
        super().__init__()
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.metrics = nn.ModuleDict(metric)
        self._losses = losses
        self.warmup_steps = int(warmup_steps or 0)
        self._warmup_scheduler = None
        self._pending_warmup = None
        self.evaluation_metrics = {}

    def on_load_checkpoint(self, checkpoint):
        for name, module in self.named_modules():
            key = f"{name}.weight"
            if isinstance(module, nn.CrossEntropyLoss) and module.weight is None and key in checkpoint["state_dict"]:
                module.weight = torch.empty_like(checkpoint["state_dict"][key])
        if self.trainer.state.fn == "fit":
            self._pending_warmup = checkpoint.get("lr_warmup")
            if "state_dict_live" in checkpoint:
                checkpoint["state_dict_ema"] = checkpoint["state_dict"]
                checkpoint["state_dict"] = checkpoint["state_dict_live"]

    def on_save_checkpoint(self, checkpoint):
        if self._warmup_scheduler is not None:
            checkpoint["lr_warmup"] = self._warmup_scheduler.state_dict()

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        groups = ([], [])
        for name, parameter in self.named_parameters():
            if parameter.grad is not None:
                groups[int("pos_bias_head" in name or "rp_pos_mlp" in name)].append(parameter)
        for parameters in groups:
            if parameters:
                nn.utils.clip_grad_norm_(parameters, gradient_clip_val)

    def training_step(self, batch, batch_idx):
        result = self.train_forward(batch=batch)
        loss = self._losses.update(result)
        if not loss.requires_grad:
            # No supervised targets in this batch: skip the step rather than let Adam
            # apply a momentum update on a zero gradient.
            return None
        auxiliary = result.get("aux_loss")
        if auxiliary is not None:
            loss = loss + auxiliary
        values = {"loss": loss, **self._losses.current}
        if auxiliary is not None:
            values["aux_rp_loss"] = auxiliary
        self.log_dict(
            {f"train/{key}": value for key, value in values.items()},
            on_step=True,
            on_epoch=False,
            batch_size=self._batch_size(batch),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        result = self.test_forward(batch=batch)
        loss = self._losses.update(result)
        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=self._batch_size(batch))
        self._update_metrics(result)

    def on_validation_epoch_end(self):
        values = {}
        for name, metric in self.metrics.named_children():
            values.update({f"val/{name}_{key}": value for key, value in metric.compute().items()})
            metric.reset()
        f1 = [value for key, value in values.items() if "macro_f1" in key.lower()]
        if f1:
            values["val/joint_macro_f1"] = torch.stack(f1).mean()
        self.log_dict(values, on_epoch=True)
        self.evaluation_metrics = {**values, "val/loss": self.trainer.callback_metrics["val/loss"]}

    def predict_step(self, batch, batch_idx):
        return self.test_forward(batch=batch)

    def configure_optimizers(self):
        optimizer = self.optimizer_partial([parameter for parameter in self.parameters() if parameter.requires_grad])
        scheduler = self.scheduler_partial(optimizer=optimizer)
        if self.warmup_steps > 0:
            self._warmup_scheduler = pytorch_warmup.LinearWarmup(optimizer, warmup_period=self.warmup_steps)
            if self._pending_warmup is not None:
                self._warmup_scheduler.load_state_dict(self._pending_warmup)
                self._pending_warmup = None
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def lr_scheduler_step(self, scheduler, metric):
        from contextlib import nullcontext

        context = self._warmup_scheduler.dampening() if self._warmup_scheduler is not None else nullcontext()
        with context:
            scheduler.step()

    def train_forward(self, batch) -> dict[str, Any]:
        return self._forward_batch(batch)

    @torch.no_grad()
    def test_forward(self, batch) -> dict[str, Any]:
        return self._forward_batch(batch)

    def _forward_batch(self, batch) -> dict[str, Any]:
        raise NotImplementedError

    def _update_metrics(self, result):
        for metric in self.metrics.children():
            metric.update(**result)

    @staticmethod
    def _batch_size(batch):
        return int(batch.num_graphs) if hasattr(batch, "num_graphs") else None
