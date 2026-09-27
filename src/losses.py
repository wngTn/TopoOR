from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.data.datasets.constants import PREDICATE_NONE_ID


def effective_number_weights(counts: Tensor, beta: float = 0.999) -> Tensor:
    """Class-balanced weights via the effective number of samples (Cui et al. 2019).

    w_c = (1 - beta) / (1 - beta^{n_c}), with absent classes (n_c=0) -> 0,
    normalised so the mean weight over PRESENT classes is 1 (keeps loss scale
    comparable to unweighted CE).
    """
    counts = counts.double()
    eff = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), counts)
    w = torch.where(counts > 0, (1.0 - beta) / eff.clamp(min=1e-12), torch.zeros_like(eff))
    present = counts > 0
    if present.any():
        w = w / w[present].mean().clamp(min=1e-12)
    return w.float()


class WorkflowRecognitionLoss(nn.Module):
    """Next Action and Robot Phase cross-entropy with Kendall-style homoscedastic uncertainty
    weighting: sum_i exp(-s_i) * L_i + 0.5 * s_i with learned s_i = log(variance_i)."""

    TASKS = {
        "NextAction_Loss": ("next_action_logits", "next_action_labels"),
        "RobotPhase_Loss": ("robot_phase_logits", "robot_phase_labels"),
    }

    def __init__(self):
        super().__init__()
        self.current = {}
        self._losses_func = nn.ModuleDict({name: nn.CrossEntropyLoss(label_smoothing=0.1) for name in self.TASKS})
        self.log_vars = nn.Parameter(torch.zeros(len(self.TASKS), dtype=torch.float32))

    def update(self, ret_val: dict[str, Any]) -> Tensor:
        self.current.clear()
        weighted = []
        for i, (name, (logits_key, labels_key)) in enumerate(self.TASKS.items()):
            loss = self._losses_func[name](ret_val[logits_key], ret_val[labels_key])
            self.current[name] = loss.detach()
            s = self.log_vars[i]
            weighted.append(torch.exp(-s) * loss + 0.5 * s)
        return torch.stack(weighted).sum()


class RelationPredictionLoss(nn.Module):
    """Factorized relation loss: relation-vs-none detection, then classification of the real predicates."""

    def __init__(self, none_weight: float = 0.1):
        super().__init__()
        self.current = {}
        self.none_weight = float(none_weight)

    def _factorized_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        real_lse = torch.logsumexp(logits[:, :PREDICATE_NONE_ID], dim=1, keepdim=True)
        none_logit = logits[:, PREDICATE_NONE_ID : PREDICATE_NONE_ID + 1]
        bin_logits = torch.cat([real_lse, none_logit], dim=1)
        is_none = (labels == PREDICATE_NONE_ID).long()
        bin_weight = logits.new_tensor([1.0, self.none_weight])
        loss = F.cross_entropy(bin_logits, is_none, weight=bin_weight)

        pos = labels != PREDICATE_NONE_ID
        if pos.any():
            loss = loss + F.cross_entropy(logits[pos, :PREDICATE_NONE_ID], labels[pos])
        return loss

    def update(self, ret_val: dict[str, Any]) -> Tensor:
        """A batch without entity pairs returns a constant zero, so the training step is skipped."""
        self.current.clear()
        logits, labels = ret_val["relation_logits"], ret_val["relation_labels"]
        if logits.size(0) == 0:
            return logits.new_zeros(())
        loss = self._factorized_loss(logits, labels)
        self.current["Relation_Loss"] = loss.detach()
        return loss
