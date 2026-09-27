from __future__ import annotations

import random

import torch
from torch_geometric.utils import subgraph

from src.data.datasets.constants import (
    HYPER_TYPE_PERSON,
    NODE_TYPE_EVIDENCE,
    NODE_TYPE_JOINT,
    R0_COL_ENTITY_ID,
    R0_COL_NODE_TYPE,
    R2_COL_HYPER_TYPE,
    R2_COL_PERSON_ENTITY_ID,
)
from src.data.temporal_graph import TemporalSuperGraph


def _remove_nodes_from_graph(bg: TemporalSuperGraph, keep_mask: torch.Tensor) -> TemporalSuperGraph:
    """
    Remove nodes where keep_mask is False, CASCADE removal up the hierarchy
    (rank-0 drop → orphaned rank-1 drop → orphaned rank-2 drop), reindex edges,
    and update all aligned tensors.

    The key insight: rank-1 cells MUST have exactly 2 rank-0 incidences.
    If either endpoint is removed, the rank-1 cell itself must be removed.
    Similarly, rank-2 cells that lose all incident rank-1 cells should be removed.
    """
    N_old = bg.x.size(0)
    assert keep_mask.shape == (N_old,), f"keep_mask shape {keep_mask.shape} != ({N_old},)"
    if keep_mask.all():
        return bg
    if bg.edge_index.size(1) > 0:
        src, dst = (bg.edge_index[0], bg.edge_index[1])
        rank_src = bg.rank[src]
        rank_dst = bg.rank[dst]
        r0_to_r1 = (rank_src == 0) & (rank_dst == 1)
        r1_to_r0 = (rank_src == 1) & (rank_dst == 0)
        incidence_01 = r0_to_r1 | r1_to_r0
        if incidence_01.any():
            inc_edges = bg.edge_index[:, incidence_01]
            r0_nodes = torch.where(bg.rank[inc_edges[0]] == 0, inc_edges[0], inc_edges[1])
            r1_nodes = torch.where(bg.rank[inc_edges[0]] == 1, inc_edges[0], inc_edges[1])
            keep_mask[r1_nodes[~keep_mask[r0_nodes]]] = False
        r1_to_r2 = (rank_src == 1) & (rank_dst == 2)
        r2_to_r1 = (rank_src == 2) & (rank_dst == 1)
        incidence_12 = r1_to_r2 | r2_to_r1
        if incidence_12.any():
            r2_mask = bg.rank == 2
            r2_indices = torch.nonzero(r2_mask, as_tuple=False).view(-1)
            if r2_indices.size(0) > 0:
                inc_edges_12 = bg.edge_index[:, incidence_12]
                r1_nodes = torch.where(bg.rank[inc_edges_12[0]] == 1, inc_edges_12[0], inc_edges_12[1])
                r2_nodes = torch.where(bg.rank[inc_edges_12[0]] == 2, inc_edges_12[0], inc_edges_12[1])
                surviving = torch.bincount(r2_nodes[keep_mask[r1_nodes]], minlength=N_old)
                keep_mask[r2_indices[surviving[r2_indices] == 0]] = False
    if keep_mask.all():
        return bg
    new_edge_index, _ = subgraph(keep_mask, bg.edge_index, relabel_nodes=True, num_nodes=N_old)
    new_bg = TemporalSuperGraph(
        x=bg.x[keep_mask],
        t=bg.t[keep_mask],
        edge_index=new_edge_index,
        rank=bg.rank[keep_mask],
        is_last_frame=bg.is_last_frame[keep_mask],
        screen_txt=getattr(bg, "screen_txt", None),
        screen_json=getattr(bg, "screen_json", None),
        screen_image=getattr(bg, "screen_image", None),
        audio=getattr(bg, "audio", None),
        images=getattr(bg, "images", None),
        cam_params=getattr(bg, "cam_params", None),
        affine=getattr(bg, "affine", None),
        next_action=bg.next_action,
        robot_phase=bg.robot_phase,
        robot_phase_per_frame=getattr(bg, "robot_phase_per_frame", None),
        relation_labels=bg.relation_labels,
        num_relation_labels=bg.num_relation_labels,
        sequence=bg.sequence,
        frame_idx=bg.frame_idx,
    )
    return new_bg


class RandomPersonDropout:
    """Drop each person with probability ``p``, keeping at least one person.

    A dropped person loses its rank-0 joint and evidence cells and its rank-2 person
    cell; ``_remove_nodes_from_graph`` then removes the cells this orphans.
    """

    def __init__(self, p: float):
        self.p = p

    def __call__(self, bg: TemporalSuperGraph) -> TemporalSuperGraph:
        device = bg.x.device
        N = bg.x.size(0)
        r0_mask = bg.rank == 0
        node_types = bg.x[:, R0_COL_NODE_TYPE].long()
        is_joint = r0_mask & (node_types == NODE_TYPE_JOINT)
        if not is_joint.any():
            return bg
        joint_entity_ids = bg.x[is_joint, R0_COL_ENTITY_ID].long()
        unique_person_ids = joint_entity_ids.unique().tolist()
        if len(unique_person_ids) <= 1:
            return bg
        num_can_drop = len(unique_person_ids) - 1
        persons_to_drop = set()
        shuffled = unique_person_ids.copy()
        random.shuffle(shuffled)
        for pid in shuffled:
            if len(persons_to_drop) >= num_can_drop:
                break
            if random.random() < self.p:
                persons_to_drop.add(pid)
        if not persons_to_drop:
            return bg
        keep_mask = torch.ones(N, dtype=torch.bool, device=device)
        entity_ids = bg.x[:, R0_COL_ENTITY_ID].long()
        for pid in persons_to_drop:
            drop_joints = r0_mask & (entity_ids == pid)
            keep_mask &= ~drop_joints
            is_evidence = r0_mask & (node_types == NODE_TYPE_EVIDENCE) & (entity_ids == pid)
            keep_mask &= ~is_evidence
            r2_mask = bg.rank == 2
            if r2_mask.any():
                hyper_types = bg.x[:, R2_COL_HYPER_TYPE].long()
                is_person_hyper = r2_mask & (hyper_types == HYPER_TYPE_PERSON)
                person_hyper_match = is_person_hyper & (bg.x[:, R2_COL_PERSON_ENTITY_ID].long() == pid)
                keep_mask &= ~person_hyper_match
        return _remove_nodes_from_graph(bg, keep_mask)


class RandomNodeDropout:
    """Drop a fraction ``p`` of the rank-0 cells, and at least one.

    ``_remove_nodes_from_graph`` then removes every rank-1 cell that lost an endpoint
    and every rank-2 cell left without a rank-1 cell.
    """

    def __init__(self, p: float):
        self.p = p

    def __call__(self, bg: TemporalSuperGraph) -> TemporalSuperGraph:
        N = bg.x.size(0)
        keep_mask = torch.ones(N, dtype=torch.bool, device=bg.x.device)
        r0_mask = bg.rank == 0
        r0_indices = torch.nonzero(r0_mask, as_tuple=False).view(-1)
        if r0_indices.size(0) == 0:
            return bg
        num_drop = max(1, int(r0_indices.size(0) * self.p))
        drop_indices = r0_indices[torch.randperm(r0_indices.size(0))[:num_drop]]
        keep_mask[drop_indices] = False
        return _remove_nodes_from_graph(bg, keep_mask)


class GraphAugmentor:
    """Person dropout followed by rank-0 cell dropout."""

    def __init__(self, person_dropout_p: float, node_dropout_p: float):
        self.augmentations = [RandomPersonDropout(person_dropout_p), RandomNodeDropout(node_dropout_p)]

    def __call__(self, bg: TemporalSuperGraph) -> TemporalSuperGraph:
        for aug in self.augmentations:
            bg = aug(bg)
        return bg
