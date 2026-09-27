from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from timm.layers.mlp import Mlp

from src.data.datasets.constants import (
    HYPER_TYPE_PERSON,
    NODE_TYPE_OBJECT,
    NUM_EDGE_TYPES,
    NUM_PREDICATES,
    PREDICATE_NONE_ID,
    R0_COL_ENTITY_ID,
    R0_COL_NODE_TYPE,
    R2_COL_HYPER_TYPE,
    R2_COL_PERSON_ENTITY_ID,
)
from src.data.temporal_graph import TemporalSuperGraph
from src.models.modules.base_module import BaseModule
from src.models.modules.encoding import _MAX_ENTITY_ID, SymmetricEndpointSummary
from src.models.modules.vision import ResNet18FeatureExtractor, SmallConvNet
from src.models.ops import ProjAttn, scatter_mean

from .encoding import CellInitialization, RelationPredictionEncoding


@dataclass
class RelationPredictionConfig:
    cell_dim: int = 64
    pool_dropout: float = field(default=0.1, init=False)


class ScalarGateFusion(nn.Module):
    def __init__(self, dim: int, num_streams: int = 3):
        super().__init__()
        self.num_streams = num_streams
        self.interact_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.norm = nn.LayerNorm(dim)

    def forward(self, *streams: torch.Tensor) -> torch.Tensor:
        assert len(streams) == self.num_streams
        additive = sum(streams)
        interact = streams[0].new_zeros(streams[0].shape)
        for i in range(len(streams)):
            for j in range(i + 1, len(streams)):
                interact = interact + streams[i] * streams[j]
        interact = self.interact_proj(interact)
        gate = torch.sigmoid(self.gate_logit)
        return self.norm(additive + gate * interact)


class RelationPredictionModule(CellInitialization, RelationPredictionEncoding, BaseModule):
    """Predict pairwise relations from the topological scene graph."""

    def __init__(self, backbone, optimizer, scheduler, metric, losses, **options):
        cfg = RelationPredictionConfig(**options)
        super().__init__(optimizer, scheduler, metric, losses)
        self.cell_dim = cfg.cell_dim
        self._init_evidence(cfg)
        self._init_rank0(cfg)
        self._init_rank1(cfg)
        self._init_person(cfg)
        self._init_triad(cfg)
        self._init_visual(cfg)
        self._init_task_heads(cfg, backbone)
        self._initialize_parameters()

    def _aggregate_entity_embeddings(
        self, node_emb: torch.Tensor, bg: TemporalSuperGraph
    ) -> tuple[list[dict[int, torch.Tensor]], int]:
        B = int(bg.batch.max().item() + 1) if bg.batch.numel() else 0
        entity_embs_per_sample: list[dict[int, torch.Tensor]] = [{} for _ in range(B)]
        r0_mask = (bg.rank == 0) & (bg.x[:, R0_COL_NODE_TYPE].long() == NODE_TYPE_OBJECT) & bg.is_last_frame
        r2_mask = (bg.rank == 2) & (bg.x[:, R2_COL_HYPER_TYPE].long() == HYPER_TYPE_PERSON) & bg.is_last_frame
        for mask, entity_column in ((r0_mask, R0_COL_ENTITY_ID), (r2_mask, R2_COL_PERSON_ENTITY_ID)):
            if not mask.any():
                continue
            entity_ids = bg.x[mask, entity_column].long()
            composite_key = bg.batch[mask] * _MAX_ENTITY_ID + entity_ids
            unique_keys, inverse = composite_key.unique(return_inverse=True)
            aggregated = scatter_mean(node_emb[mask], inverse, dim_size=unique_keys.size(0))
            for k_idx, key in enumerate(unique_keys.tolist()):
                b = key // _MAX_ENTITY_ID
                eid = key % _MAX_ENTITY_ID
                entity_embs_per_sample[b][eid] = aggregated[k_idx]
        return (entity_embs_per_sample, B)

    def _build_relation_targets(
        self, bg: TemporalSuperGraph, entity_embs_per_sample: list[dict[int, torch.Tensor]], B: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = bg.x.device
        num_rels = bg.num_relation_labels
        rel_labels_all = bg.relation_labels
        offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), num_rels.cumsum(0)])
        all_pair_embs = []
        all_pair_labels = []
        pair_dim: int = self.backbone.out_features * 4
        for b in range(B):
            entity_dict = entity_embs_per_sample[b]
            start, end = (int(offsets[b].item()), int(offsets[b + 1].item()))
            gt_predicate: dict[tuple[int, int], int] = {}
            gt_entities: set[int] = set()
            for row in rel_labels_all[start:end]:
                eid_i, pred_id, eid_j = (int(row[0]), int(row[1]), int(row[2]))
                if eid_i in entity_dict and eid_j in entity_dict:
                    gt_predicate[eid_i, eid_j] = pred_id
                    gt_entities.add(eid_i)
                    gt_entities.add(eid_j)
            universe = sorted(gt_entities)
            for eid_i in universe:
                emb_i = entity_dict[eid_i]
                for eid_j in universe:
                    if eid_i == eid_j:
                        continue
                    emb_j = entity_dict[eid_j]
                    pred_id = gt_predicate.get((eid_i, eid_j), PREDICATE_NONE_ID)
                    pair_feat = torch.cat([emb_i, emb_j, emb_i * emb_j, (emb_i - emb_j).abs()])
                    all_pair_embs.append(pair_feat)
                    all_pair_labels.append(pred_id)
        if not all_pair_embs:
            return (torch.zeros((0, pair_dim), device=device), torch.zeros((0,), dtype=torch.long, device=device))
        pair_features = torch.stack(all_pair_embs)
        pair_labels = torch.tensor(all_pair_labels, dtype=torch.long, device=device)
        return (pair_features, pair_labels)

    def _forward_batch(self, bg: TemporalSuperGraph) -> dict[str, Any]:
        bg = bg.to(self.device)
        x = self._encode_cells(bg)
        node_emb = self.backbone(x, bg.edge_index, bg.rank, t=bg.t, batch_index=bg.batch)
        entity_embs_per_sample, B = self._aggregate_entity_embeddings(node_emb, bg)
        pair_features, pair_labels = self._build_relation_targets(bg, entity_embs_per_sample, B)
        if pair_features.size(0) > 0:
            relation_logits = self.relation_head(pair_features)
        else:
            relation_logits = torch.zeros((0, NUM_PREDICATES), device=self.device)
        return {"sequences": bg.sequence, "relation_logits": relation_logits, "relation_labels": pair_labels}

    def _init_evidence(self, cfg):
        self.proj_attn = ProjAttn(d_model=cfg.cell_dim, n_heads=4, n_points=4)
        self.view_gate = nn.Sequential(
            nn.LayerNorm(cfg.cell_dim),
            nn.Linear(cfg.cell_dim, cfg.cell_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.cell_dim // 2, 1),
        )
        self.screen_proj_txt = nn.Linear(768, cfg.cell_dim)
        self.screen_proj_json = nn.Linear(768, cfg.cell_dim)
        self.audio_proj = nn.Linear(512, cfg.cell_dim)

    def _init_rank1(self, cfg):
        self.r1_geo_proj = nn.Sequential(nn.Linear(4, cfg.cell_dim), nn.GELU(), nn.LayerNorm(cfg.cell_dim))
        self.edge_type_emb = nn.Embedding(NUM_EDGE_TYPES, cfg.cell_dim)
        self.r1_endpoint_summary = SymmetricEndpointSummary(in_dim=cfg.cell_dim, out_dim=cfg.cell_dim)
        self.r1_fuse = ScalarGateFusion(dim=cfg.cell_dim, num_streams=3)

    def _init_visual(self, cfg):
        self.visual_backbone = ResNet18FeatureExtractor(frozen=True)
        self.screen_backbone = SmallConvNet(out_dim=cfg.cell_dim)

    def _init_task_heads(self, cfg, backbone):
        self.cell_ln = nn.LayerNorm(cfg.cell_dim)
        self.backbone = backbone
        pair_input_dim: int = self.backbone.out_features * 4
        self.relation_head = Mlp(
            pair_input_dim, hidden_features=256, out_features=NUM_PREDICATES, norm_layer=nn.LayerNorm
        )

    def _initialize_parameters(self):
        for name, module in self.named_modules():
            if name.split(".")[0] == "backbone":
                continue
            self._init_weights(module)
