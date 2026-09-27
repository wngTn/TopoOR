import torch
from torchmetrics import Metric
from torchmetrics.classification import MulticlassStatScores

from src.data.datasets.constants import NUM_NEXT_ACTION_CLASSES, NUM_PREDICATES, NUM_ROBOT_PHASE_CLASSES


def present_class_metrics(stats):
    tp, fp, _, fn, support = stats.to(torch.float64).unbind(-1)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    active = support > 0
    macro = (f1 * active).sum() / active.sum().clamp_min(1)
    accuracy = tp.sum() / support.sum().clamp_min(1)
    return macro, accuracy


class WorkflowRecognitionMetrics(Metric):
    def __init__(
        self, num_next_action_classes=NUM_NEXT_ACTION_CLASSES, num_robot_phase_classes=NUM_ROBOT_PHASE_CLASSES
    ):
        super().__init__()
        self.next_action = MulticlassStatScores(num_next_action_classes, average=None)
        self.robot_phase = MulticlassStatScores(num_robot_phase_classes, average=None)

    def update(self, sequences, next_action_logits, robot_phase_logits, next_action_labels, robot_phase_labels, **_):
        self.next_action.update(next_action_logits, next_action_labels)
        selected = [index for index, sequence in enumerate(sequences) if sequence != "004_PKA"]
        if selected:
            self.robot_phase.update(robot_phase_logits[selected], robot_phase_labels[selected])

    def compute(self):
        return {
            f"{name}_macro_f1": present_class_metrics(getattr(self, name).compute())[0]
            for name in ("next_action", "robot_phase")
        }

    def reset(self):
        super().reset()
        self.next_action.reset()
        self.robot_phase.reset()

class RelationPredictionMetrics(Metric):
    def __init__(self, num_predicates=NUM_PREDICATES):
        super().__init__()
        self.relations = MulticlassStatScores(num_predicates, average=None)

    def update(self, relation_logits, relation_labels, **_):
        if relation_labels.numel():
            self.relations.update(relation_logits, relation_labels)

    def compute(self):
        f1, accuracy = present_class_metrics(self.relations.compute())
        return {"relation_macro_f1": f1, "relation_accuracy": accuracy}

    def reset(self):
        super().reset()
        self.relations.reset()
