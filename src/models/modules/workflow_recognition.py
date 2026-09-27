import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.data.datasets.constants import NUM_EDGE_TYPES, NUM_NEXT_ACTION_CLASSES, NUM_ROBOT_PHASE_CLASSES
from src.data.temporal_graph import TemporalSuperGraph
from src.losses import effective_number_weights
from src.models.modules.base_module import BaseModule
from src.models.modules.encoding import SymmetricEndpointSummary
from src.models.modules.vision import ResNet18FeatureExtractor, SmallConvNet
from src.models.ops import ProjAttn, scatter_add, scatter_mean, scatter_softmax

from .encoding import CellInitialization, WorkflowRecognitionEncoding

log = logging.getLogger(__name__)

ACTION_LAST_K_FRAMES = 6
PHASE_LAST_K_FRAMES = 3
AUX_WEIGHT = 0.5
REFINE_WEIGHT = 0.5
POS_NUM_FREQS = 6
POS_DIM = 32
POS_NORM = 2048.0  # causal frame-position scale, independent of take length


@dataclass
class WorkflowRecognitionConfig:
    cell_dim: int = 64
    enc_dropout: float = 0.15
    head_dropout: float = 0.3
    pool_dropout: float = 0.2
    warmup_steps: int = 0


def last_frames_mask(t: torch.Tensor, last_k: int) -> torch.Tensor:
    """Select the cells in the last ``last_k`` frames of the window."""
    window = int(t.max().item() + 1) if t.numel() else 0
    return t >= max(0, window - last_k)


class ProjectedGateFusion(nn.Module):
    def __init__(self, dim: int, num_streams: int = 3, dropout: float = 0.0):
        super().__init__()
        self.num_streams = num_streams
        self.interact_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.gate_proj = nn.Linear(dim, 1)
        nn.init.constant_(self.gate_proj.bias, -2.0)
        nn.init.zeros_(self.gate_proj.weight)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, *streams: torch.Tensor) -> torch.Tensor:
        assert len(streams) == self.num_streams
        additive = sum(streams)
        interact = streams[0].new_zeros(streams[0].shape)
        for i in range(len(streams)):
            for j in range(i + 1, len(streams)):
                interact = interact + streams[i] * streams[j]
        interact = self.interact_proj(interact)
        gate = torch.sigmoid(self.gate_proj(additive))
        return self.drop(self.norm(additive + gate * interact))


class QueryAttentionPool(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.key = nn.Linear(dim, dim, bias=False)
        self.query = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, batch: torch.Tensor, dim_size: int | None = None) -> torch.Tensor:
        k = self.key(x)
        logits = (k * self.query).sum(dim=-1)
        alpha = scatter_softmax(logits, batch)
        alpha = self.attn_drop(alpha)
        out = x * alpha.unsqueeze(-1)
        return scatter_add(out, batch, dim_size=dim_size)


class RankWisePool(nn.Module):
    """Pool each task's cells rank by rank and mix the ranks with a learned gate and per-rank prior."""

    NUM_RANKS = 3
    RANK_PRIORS = {"action": [0.01, 0.1, 0.2], "phase": [0.01, 0.0, 0.3]}

    def __init__(self, dim: int, pool_dropout: float = 0.2):
        super().__init__()
        R = self.NUM_RANKS
        self.pools = nn.ModuleDict(
            {
                task: nn.ModuleList([QueryAttentionPool(dim, dropout=pool_dropout) for _ in range(R)])
                for task in self.RANK_PRIORS
            }
        )
        self.rank_gate = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.LayerNorm(R * dim + R),
                    nn.Linear(R * dim + R, dim),
                    nn.GELU(),
                    nn.Linear(dim, R),
                )
                for task in self.RANK_PRIORS
            }
        )
        self.rank_prior = nn.ParameterDict()
        for task, prior in self.RANK_PRIORS.items():
            self.rank_prior[task] = nn.Parameter(torch.tensor(prior))

    def _pool_one_task(self, task: str, x: torch.Tensor, batch: torch.Tensor, rank: torch.Tensor, mask: torch.Tensor):
        x_m = x[mask]
        b_m = batch[mask]
        r_m = rank[mask]
        if b_m.numel() == 0:
            return x.new_zeros((0, x.size(-1)))
        B = int(b_m.max().item() + 1)
        D = x.size(-1)
        R = self.NUM_RANKS
        rank_outputs = []
        rank_present = []
        rank_counts = []
        for k in range(R):
            mk = r_m == k
            cnt_k = scatter_add(mk.float(), b_m, dim_size=B)
            rank_counts.append(cnt_k)
            pres_k = cnt_k > 0
            rank_present.append(pres_k)
            if mk.any():
                rank_outputs.append(self.pools[task][k](x_m[mk], b_m[mk], dim_size=B))
            else:
                rank_outputs.append(x.new_zeros((B, D)))
        pooled_cat = torch.cat(rank_outputs, dim=-1)
        counts = torch.stack(rank_counts, dim=-1)
        counts_feat = torch.log1p(counts)
        gate_in = torch.cat([pooled_cat, counts_feat], dim=-1)
        logits = self.rank_gate[task](gate_in)
        logits = logits + self.rank_prior[task].unsqueeze(0)
        present = torch.stack(rank_present, dim=-1)
        logits = logits.masked_fill(~present, float("-inf"))
        all_missing = ~present.any(dim=-1, keepdim=True)
        logits = torch.where(all_missing, torch.zeros_like(logits), logits)
        w = torch.softmax(logits, dim=-1)
        pooled_stack = torch.stack(rank_outputs, dim=1)
        out = (w.unsqueeze(-1) * pooled_stack).sum(dim=1)
        return out

    def forward(self, x: torch.Tensor, batch: torch.Tensor, rank: torch.Tensor, task_masks: dict[str, torch.Tensor]):
        outs: dict[str, torch.Tensor] = {}
        for task in self.RANK_PRIORS:
            outs[task] = self._pool_one_task(task, x, batch, rank, task_masks[task])
        return outs


class WorkflowRecognitionModule(CellInitialization, WorkflowRecognitionEncoding, BaseModule):
    def __init__(self, backbone, optimizer, scheduler, metric, losses, **options):
        cfg = WorkflowRecognitionConfig(**options)
        super().__init__(optimizer, scheduler, metric, losses, warmup_steps=cfg.warmup_steps)
        self._init_options(cfg)
        self._init_evidence(cfg)
        self._init_rank0(cfg)
        self._init_rank1(cfg)
        self._init_person(cfg)
        self._init_triad(cfg)
        self._init_visual(cfg)
        self._init_task_heads(cfg, backbone)
        self._init_position()
        self._init_auxiliary(cfg)
        self._initialize_parameters()

    def _rp_pos_encode(self, take_pos: torch.Tensor) -> torch.Tensor:
        """Encode causal frame position using fixed, not take-length, normalization."""
        p = (take_pos.float() / POS_NORM).unsqueeze(-1)
        ang = p * self.rp_pos_freqs
        feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.rp_pos_mlp(feat)

    def _forward_batch(self, bg: TemporalSuperGraph) -> dict[str, Any]:
        bg = bg.to(self.device)
        x = self._encode_cells(bg)
        node_emb = self.backbone(x, bg.edge_index, bg.rank, t=bg.t, batch_index=bg.batch)
        task_masks = {
            "action": last_frames_mask(bg.t, ACTION_LAST_K_FRAMES),
            "phase": last_frames_mask(bg.t, PHASE_LAST_K_FRAMES),
        }
        pooled = self.task_pool(node_emb, bg.batch, rank=bg.rank, task_masks=task_masks)
        action_input = pooled["action"] + self.action_adapter(pooled["action"])
        phase_input = pooled["phase"] + self.phase_adapter(pooled["phase"])
        base_phase_logits = self.robot_phase_head(phase_input)
        pos_bias = self.pos_bias_head(self._rp_pos_encode(bg.take_pos.to(self.device)))
        refine_logits = base_phase_logits.detach() + pos_bias
        robot_phase_logits = base_phase_logits if self.training else refine_logits
        retval = {
            "sequences": bg.sequence,
            "frame_idx": bg.frame_idx,
            "next_action_logits": self.next_action_head(action_input),
            "robot_phase_logits": robot_phase_logits,
            "next_action_labels": bg.next_action.to(self.device),
            "robot_phase_labels": bg.robot_phase.to(self.device),
        }
        if self.training:
            T_win = int(bg.t.max().item()) + 1
            B = int(bg.batch.max().item()) + 1
            per_frame_emb = scatter_mean(node_emb, bg.batch * T_win + bg.t, dim_size=B * T_win)
            pf_logits = self.robot_phase_aux_head(per_frame_emb)
            pf_labels = bg.robot_phase_per_frame.to(self.device).long()
            per_frame = AUX_WEIGHT * self.robot_phase_aux_ce(pf_logits, pf_labels)
            refine = self.robot_phase_refine_ce(refine_logits, bg.robot_phase.to(self.device))
            retval["aux_loss"] = per_frame + REFINE_WEIGHT * refine
        return retval

    def setup(self, stage: str) -> None:
        """Install loss-weight buffers before EMA copies the module."""
        if stage == "fit":
            self._setup_class_balance()

    def _setup_class_balance(self) -> None:
        """Weight every cross-entropy by the training windows' class frequencies."""
        na_counts, rp_counts = self._count_train_window_labels(self.trainer.datamodule.train_dataset)
        na_weight, rp_weight = effective_number_weights(na_counts), effective_number_weights(rp_counts)
        self._losses._losses_func["NextAction_Loss"].weight = na_weight
        self._losses._losses_func["RobotPhase_Loss"].weight = rp_weight
        self.robot_phase_aux_ce.weight = rp_weight
        self.robot_phase_refine_ce.weight = rp_weight
        for task, counts, weight in (("Next Action", na_counts, na_weight), ("Robot Phase", rp_counts, rp_weight)):
            weights = [round(w, 2) for w in weight.tolist()]
            log.info("Class-balanced %s: counts=%s -> weights=%s", task, counts.tolist(), weights)

    @staticmethod
    def _count_train_window_labels(ds) -> tuple[torch.Tensor, torch.Tensor]:
        """Count labels at each training window's final, clamped frame."""
        na = torch.zeros(NUM_NEXT_ACTION_CLASSES, dtype=torch.long)
        rp = torch.zeros(NUM_ROBOT_PHASE_CLASSES, dtype=torch.long)
        W = ds.temporal_window
        for sequence, start_idx in ds.samples:
            frames = ds.sequence_frames[sequence]
            fm = frames[min(max(start_idx + W - 1, 0), len(frames) - 1)]
            na[int(fm["next_action"])] += 1
            rp[int(fm["robot_phase"])] += 1
        return (na, rp)

    def _init_options(self, cfg):
        self.cell_dim = cfg.cell_dim
        self.r0_drop = nn.Dropout(cfg.enc_dropout)
        self.r2_drop = nn.Dropout(cfg.enc_dropout)
        self.evidence_drop = nn.Dropout(cfg.enc_dropout)

    def _init_evidence(self, cfg):
        self.proj_attn = ProjAttn(d_model=cfg.cell_dim, n_heads=4, n_points=4)
        self.screen_proj_txt = nn.Linear(768, cfg.cell_dim)
        self.screen_proj_json = nn.Linear(768, cfg.cell_dim)
        self.audio_proj = nn.Linear(512, cfg.cell_dim)

    def _init_rank1(self, cfg):
        self.r1_geo_proj = nn.Sequential(nn.Linear(4, cfg.cell_dim), nn.GELU(), nn.LayerNorm(cfg.cell_dim))
        self.edge_type_emb = nn.Embedding(NUM_EDGE_TYPES, cfg.cell_dim)
        self.r1_endpoint_summary = SymmetricEndpointSummary(in_dim=cfg.cell_dim, out_dim=cfg.cell_dim)
        self.r1_fuse = ProjectedGateFusion(dim=cfg.cell_dim, num_streams=3, dropout=cfg.enc_dropout)

    def _init_visual(self, cfg):
        self.visual_backbone = ResNet18FeatureExtractor(frozen=False)
        self.view_gate = nn.Sequential(
            nn.LayerNorm(cfg.cell_dim),
            nn.Linear(cfg.cell_dim, cfg.cell_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.cell_dim // 2, 1),
        )
        self.screen_backbone = SmallConvNet(out_dim=cfg.cell_dim)

    def _init_task_heads(self, cfg, backbone):
        self.cell_ln = nn.LayerNorm(cfg.cell_dim)
        self.backbone = backbone
        out_features: int = self.backbone.out_features
        self.task_pool = RankWisePool(dim=out_features, pool_dropout=cfg.pool_dropout)
        adapter_dim = out_features
        self.action_adapter = nn.Sequential(
            nn.LayerNorm(out_features),
            nn.Linear(out_features, adapter_dim),
            nn.GELU(),
            nn.Dropout(cfg.enc_dropout),
            nn.Linear(adapter_dim, out_features),
            nn.Dropout(cfg.enc_dropout),
        )
        self.phase_adapter = nn.Sequential(
            nn.LayerNorm(out_features),
            nn.Linear(out_features, adapter_dim),
            nn.GELU(),
            nn.Dropout(cfg.enc_dropout),
            nn.Linear(adapter_dim, out_features),
            nn.Dropout(cfg.enc_dropout),
        )
        action_input_dim = out_features
        self.next_action_head = nn.Sequential(
            nn.LayerNorm(action_input_dim),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(action_input_dim, NUM_NEXT_ACTION_CLASSES),
        )
        self.robot_phase_head = nn.Sequential(
            nn.LayerNorm(out_features),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(out_features, NUM_ROBOT_PHASE_CLASSES),
        )

    def _init_position(self):
        freqs = torch.pow(2.0, torch.arange(POS_NUM_FREQS, dtype=torch.float32)) * torch.pi
        self.register_buffer("rp_pos_freqs", freqs, persistent=False)
        self.rp_pos_mlp = nn.Sequential(
            nn.Linear(2 * POS_NUM_FREQS, POS_DIM),
            nn.GELU(),
            nn.Linear(POS_DIM, POS_DIM),
        )
        self.pos_bias_head = nn.Linear(POS_DIM, NUM_ROBOT_PHASE_CLASSES)
        self.robot_phase_refine_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def _init_auxiliary(self, cfg):
        out_features = self.backbone.out_features
        self.robot_phase_aux_head = nn.Sequential(
            nn.LayerNorm(out_features),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(out_features, NUM_ROBOT_PHASE_CLASSES),
        )
        self.robot_phase_aux_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def _initialize_parameters(self):
        for name, module in self.named_modules():
            if name.split(".")[0] == "backbone":
                continue
            self._init_weights(module)
        with torch.no_grad():
            nn.init.zeros_(self.action_adapter[4].weight)
            nn.init.zeros_(self.action_adapter[4].bias)
            nn.init.zeros_(self.phase_adapter[4].weight)
            nn.init.zeros_(self.phase_adapter[4].bias)
            nn.init.zeros_(self.pos_bias_head.weight)
            nn.init.zeros_(self.pos_bias_head.bias)
