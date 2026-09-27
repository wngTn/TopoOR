from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from src.data.datasets.constants import (
    EDGE_TYPE_EVIDENCE_AUDIO,
    EDGE_TYPE_EVIDENCE_JSON,
    EDGE_TYPE_EVIDENCE_SCREEN_IMAGE,
    EDGE_TYPE_EVIDENCE_TEXT,
    EDGE_TYPE_EVIDENCE_VISUAL,
    EDGE_TYPE_SEMANTIC,
    EDGE_TYPE_SKELETON,
    EDGE_TYPE_SPATIAL,
    EVIDENCE_SUBTYPE_AUDIO,
    EVIDENCE_SUBTYPE_JSON,
    EVIDENCE_SUBTYPE_SCREEN_IMAGE,
    EVIDENCE_SUBTYPE_TEXT,
    HYPER_TYPE_PERSON,
    HYPER_TYPE_TRIAD,
    NODE_TYPE_EVIDENCE,
    NODE_TYPE_JOINT,
    R0_COL_ENTITY_ID,
    R0_COL_JOINT_TYPE,
    R0_COL_NODE_TYPE,
    R1_COL_EDGE_TYPE,
    R2_COL_HYPER_TYPE,
    R2_COL_PERSON_ENTITY_ID,
)


class TemporalSuperGraph(Data):
    """PyG-compatible Data object for a temporal super-graph."""


def _cell_features(frame: Data) -> torch.Tensor:
    """Stack ``x_0``/``x_1``/``x_2`` rank-major into one ``[N, feat_dim]`` tensor.

    Each rank is right-padded with zeros to the widest rank. Cell rank is carried
    separately in ``rank``.
    """
    cells = (frame.x_0, frame.x_1, frame.x_2)
    feat_dim = max(x.size(-1) for x in cells)
    return torch.cat([F.pad(x, (0, feat_dim - x.size(-1))) for x in cells])


def _should_temporally_link_rank0_key(key: tuple[int, int, int]) -> bool:
    node_type, entity_id, extra = key
    if node_type != NODE_TYPE_EVIDENCE:
        return True

    subtype = extra
    return subtype in (
        EVIDENCE_SUBTYPE_TEXT,
        EVIDENCE_SUBTYPE_JSON,
        EVIDENCE_SUBTYPE_AUDIO,
        EVIDENCE_SUBTYPE_SCREEN_IMAGE,
    )


def _rank0_identity_keys(frame: Data) -> list[tuple[int, int, int]]:
    """Return (node_type, entity_id, joint_type/subtype) in cell order."""
    columns = [R0_COL_NODE_TYPE, R0_COL_ENTITY_ID, R0_COL_JOINT_TYPE]
    rows = frame.x_0[:, columns].detach().cpu().tolist()
    return [tuple(map(int, row)) for row in rows]


def _hasse_incidence_maps(
    frame: Data,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Decode rank-local downward incidence, preserving edge order and duplicates."""
    n0, n1, n2 = frame.x_0.size(0), frame.x_1.size(0), frame.x_2.size(0)
    r1_to_r0: dict[int, list[int]] = defaultdict(list)
    r2_to_r1: dict[int, list[int]] = defaultdict(list)
    if frame.edge_index_hasse.numel() == 0:
        return r1_to_r0, r2_to_r1

    hasse_cpu = frame.edge_index_hasse.detach().cpu()
    for u, v in zip(hasse_cpu[0].tolist(), hasse_cpu[1].tolist()):
        # rank-0 ↔ rank-1 (one endpoint in [0, n0), the other in [n0, n0+n1))
        if 0 <= u < n0 and n0 <= v < n0 + n1:
            r1_to_r0[v - n0].append(u)
        elif n0 <= u < n0 + n1 and 0 <= v < n0:
            r1_to_r0[u - n0].append(v)
        # rank-1 ↔ rank-2 (one endpoint in [n0, n0+n1), the other in [n0+n1, n0+n1+n2))
        if n0 <= u < n0 + n1 and n0 + n1 <= v < n0 + n1 + n2:
            r2_to_r1[v - n0 - n1].append(u - n0)
        elif n0 + n1 <= u < n0 + n1 + n2 and n0 <= v < n0 + n1:
            r2_to_r1[u - n0 - n1].append(v - n0)
    return r1_to_r0, r2_to_r1


def _rank2_identity_keys(frame: Data, r0_keys: list[tuple], incidence) -> list[Optional[tuple]]:
    """Identify people by entity ID and triads by their constituent entities."""
    n2 = frame.x_2.size(0)
    if n2 == 0:
        return []

    columns = [R2_COL_HYPER_TYPE, R2_COL_PERSON_ENTITY_ID]
    rows = frame.x_2[:, columns].detach().cpu().tolist()
    keys: list[Optional[tuple]] = []
    r1_to_r0, r2_to_r1 = incidence
    for i, (hyper_type, entity_id) in enumerate(rows):
        hyper_type = int(hyper_type)

        if hyper_type == HYPER_TYPE_PERSON:
            keys.append(("rank2_person", int(entity_id)))

        elif hyper_type == HYPER_TYPE_TRIAD:
            incident_r1 = r2_to_r1.get(i, [])
            entity_ids = set()
            for r1_idx in incident_r1:
                for r0_idx in r1_to_r0.get(r1_idx, []):
                    entity_ids.add(r0_keys[r0_idx][1])

            if len(entity_ids) >= 2:
                keys.append(("rank2_triad", tuple(sorted(entity_ids))))
            else:
                keys.append(None)

        else:
            keys.append(None)

    return keys


def _get_rank1_identity_keys(frame: Data, r0_keys: list[tuple], incidence) -> list[Optional[tuple]]:
    """Generate stable temporal keys for rank-1 cells using incidence (Hasse) to rank-0 endpoints."""
    n1 = frame.x_1.size(0)
    if n1 == 0:
        return []

    r1_to_r0, _ = incidence

    keys: list[Optional[tuple]] = []
    edge_types = frame.x_1[:, R1_COL_EDGE_TYPE].detach().cpu().tolist()

    for i, edge_type in enumerate(edge_types):
        edge_type = int(edge_type)

        if edge_type in (
            EDGE_TYPE_EVIDENCE_VISUAL,
            EDGE_TYPE_EVIDENCE_TEXT,
            EDGE_TYPE_EVIDENCE_JSON,
            EDGE_TYPE_EVIDENCE_AUDIO,
            EDGE_TYPE_EVIDENCE_SCREEN_IMAGE,
        ):
            keys.append(None)
            continue

        incident = list(dict.fromkeys(r1_to_r0.get(i, [])))
        if not incident:
            keys.append(None)
            continue

        incident_keys = [r0_keys[idx] for idx in incident]

        if edge_type == EDGE_TYPE_SKELETON:
            entity_ids = set(k[1] for k in incident_keys)
            person_id = entity_ids.pop() if len(entity_ids) == 1 else -1
            joint_types = sorted(set(k[2] for k in incident_keys if k[2] != -1))
            if len(joint_types) == 2:
                keys.append(("rank1", EDGE_TYPE_SKELETON, person_id, joint_types[0], joint_types[1]))
            else:
                keys.append(None)

        elif edge_type in (EDGE_TYPE_SEMANTIC, EDGE_TYPE_SPATIAL):
            unique_entities = sorted(set(k[1] for k in incident_keys))
            joints = tuple(sorted(set(k[2] for k in incident_keys if k[0] == NODE_TYPE_JOINT)))

            if len(unique_entities) == 2:
                keys.append(("rank1", edge_type, unique_entities[0], unique_entities[1], joints))
            elif len(unique_entities) == 1:
                keys.append(("rank1", edge_type, unique_entities[0], unique_entities[0], joints))
            else:
                keys.append(("rank1", edge_type, tuple(unique_entities), joints))

        else:
            unique_entities = sorted(set(k[1] for k in incident_keys))
            joints = tuple(sorted(set(k[2] for k in incident_keys if k[0] == NODE_TYPE_JOINT)))
            keys.append(("rank1", edge_type, tuple(unique_entities), joints))

    return keys


def _frame_identity_map(frame, offset):
    keys = {}
    r0_keys = _rank0_identity_keys(frame)
    incidence = _hasse_incidence_maps(frame)
    rank_keys = [
        [key if _should_temporally_link_rank0_key(key) else None for key in r0_keys],
        _get_rank1_identity_keys(frame, r0_keys, incidence),
        _rank2_identity_keys(frame, r0_keys, incidence),
    ]
    for rank in rank_keys:
        for key in rank:
            if key is not None:
                keys.setdefault(key, []).append(offset)
            offset += 1
    return keys


def _temporal_edges(key_maps, device):
    edges = []
    for source, target in zip(key_maps, key_maps[1:]):
        src, dst = [], []
        for key, nodes in source.items():
            matches = target.get(key, [])
            if len(nodes) == len(matches) == 1:
                src.extend([nodes[0], matches[0]])
                dst.extend([matches[0], nodes[0]])
        if src:
            edges.append(torch.tensor([src, dst], dtype=torch.long, device=device))
    return edges


def build_causal_super_graph(window_frames):
    """Join observed frames; categorical relation labels are never shifted."""
    if not window_frames:
        raise ValueError("Window frames list is empty.")
    device = window_frames[0].x_0.device
    features, ranks, times, edges, key_maps = [], [], [], [], []
    offset = 0
    for time, frame in enumerate(window_frames):
        x = _cell_features(frame)
        features.append(x)
        ranks.append(
            torch.cat(
                [
                    torch.full((cells.size(0),), rank, dtype=torch.long, device=device)
                    for rank, cells in enumerate((frame.x_0, frame.x_1, frame.x_2))
                ]
            )
        )
        times.append(torch.full((x.size(0),), time, dtype=torch.long, device=device))
        if frame.edge_index_hasse.numel():
            edges.append(frame.edge_index_hasse.to(device) + offset)
        key_maps.append(_frame_identity_map(frame, offset))
        offset += x.size(0)
    edges.extend(_temporal_edges(key_maps, device))
    time = torch.cat(times)
    evidence = {}
    for name in ("screen_txt", "screen_json", "screen_image", "audio", "images", "cam_params", "affine"):
        values = [getattr(frame, name, None) for frame in window_frames]
        values = [value for value in values if value is not None]
        evidence[name] = torch.cat(values) if values else None
    last = window_frames[-1]
    labels = {name: getattr(last, name) for name in ("next_action", "robot_phase", "sequence", "frame_idx")}
    return TemporalSuperGraph(
        x=torch.cat(features),
        rank=torch.cat(ranks),
        t=time,
        edge_index=torch.cat(edges, dim=1) if edges else torch.zeros((2, 0), dtype=torch.long, device=device),
        is_last_frame=time == len(window_frames) - 1,
        robot_phase_per_frame=torch.cat([frame.robot_phase.reshape(1) for frame in window_frames]),
        relation_labels=last.relation_labels,
        num_relation_labels=last.num_relation_labels,
        **labels,
        **evidence,
    )
