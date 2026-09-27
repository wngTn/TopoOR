"""Per-frame combinatorial complex of an MM-OR frame.

Rank 0 holds person joints, objects and evidence cells (a visual cell per joint, the
robot's screen text/JSON/image, one room-level audio cell). Rank 1 holds skeleton,
semantic, spatial, evidence and audio-broadcast edges. Rank 2 holds one cell per person
(its skeleton edges), one per functional triad (its three pairwise edges) and one for
audio (the broadcast edges).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np
import torch
from torch_geometric.data import Data

from .constants import (
    CONF_THRESHOLD,
    EDGE_TYPE_AUDIO_BROADCAST,
    EDGE_TYPE_EVIDENCE_AUDIO,
    EDGE_TYPE_EVIDENCE_JSON,
    EDGE_TYPE_EVIDENCE_SCREEN_IMAGE,
    EDGE_TYPE_EVIDENCE_TEXT,
    EDGE_TYPE_EVIDENCE_VISUAL,
    EDGE_TYPE_SEMANTIC,
    EDGE_TYPE_SKELETON,
    EDGE_TYPE_SPATIAL,
    ENTITY_MAKO_ROBOT,
    EVIDENCE_SUBTYPE_AUDIO,
    EVIDENCE_SUBTYPE_JSON,
    EVIDENCE_SUBTYPE_SCREEN_IMAGE,
    EVIDENCE_SUBTYPE_TEXT,
    EVIDENCE_SUBTYPE_VISUAL,
    FUNCTIONAL_TRIADS,
    HEAD_GRAPH_INDEX,
    HUMAN_ENTITIES,
    HYPER_TYPE_AUDIO,
    HYPER_TYPE_PERSON,
    HYPER_TYPE_TRIAD,
    NODE_TYPE_EVIDENCE,
    NODE_TYPE_JOINT,
    NODE_TYPE_OBJECT,
    NUM_JOINTS,
    OR_CENTER,
    OR_HALF_SIZE,
    PELVIS_IDX,
    R0_COL_ENTITY_ID,
    R0_COL_JOINT_TYPE,
    R0_COL_NODE_TYPE,
    RANK0_FEAT_DIM,
    RANK1_FEAT_DIM,
    RANK2_FEAT_DIM,
    SEMANTIC_EDGES,
    SEMANTIC_THRESHOLD_MM,
    SKELETON_EDGES,
    SPATIAL_THRESHOLD_MM,
    UPPER_BODY_INDICES,
    WRIST_GRAPH_INDICES,
    WRIST_RAW_MAP,
    object_name_to_id,
    predicate_name_to_id,
)
from .geometry import _obb_surface_distance, _point_to_obb_distance

SEMANTIC_PAIRS = set(SEMANTIC_EDGES) | {(b, a) for a, b in SEMANTIC_EDGES}
ROBOT_EVIDENCE_SUBTYPES = (EVIDENCE_SUBTYPE_TEXT, EVIDENCE_SUBTYPE_JSON, EVIDENCE_SUBTYPE_SCREEN_IMAGE)
EVIDENCE_EDGE_TYPES = {
    EVIDENCE_SUBTYPE_VISUAL: EDGE_TYPE_EVIDENCE_VISUAL,
    EVIDENCE_SUBTYPE_TEXT: EDGE_TYPE_EVIDENCE_TEXT,
    EVIDENCE_SUBTYPE_JSON: EDGE_TYPE_EVIDENCE_JSON,
    EVIDENCE_SUBTYPE_AUDIO: EDGE_TYPE_EVIDENCE_AUDIO,
    EVIDENCE_SUBTYPE_SCREEN_IMAGE: EDGE_TYPE_EVIDENCE_SCREEN_IMAGE,
}


@dataclass
class GraphBuildState:
    entities: dict = field(default_factory=dict)
    entity_ids_present: set = field(default_factory=set)
    entity_to_node_ids: dict = field(default_factory=dict)
    rank0_features: list[np.ndarray] = field(default_factory=list)
    rank0_world: list[np.ndarray] = field(default_factory=list)
    rank1_features: list[np.ndarray] = field(default_factory=list)
    rank2_features: list[np.ndarray] = field(default_factory=list)
    inc_0_1_src: list[int] = field(default_factory=list)
    inc_0_1_tgt: list[int] = field(default_factory=list)
    inc_1_2_src: list[int] = field(default_factory=list)
    inc_1_2_tgt: list[int] = field(default_factory=list)
    physical_to_evidence_list: defaultdict = field(default_factory=lambda: defaultdict(list))
    person_to_skel_edge_cells: defaultdict = field(default_factory=lambda: defaultdict(list))
    inter_entity_best_edge: dict = field(default_factory=dict)
    audio_edge_cells: list[int] = field(default_factory=list)
    audio_node_idx: int = -1
    n0: int = 0
    n1: int = 0
    n2: int = 0


def build_scene_graph(bbox_data: dict, pose_data: list[dict], frame_meta: dict) -> Data:
    state = GraphBuildState()
    _collect_entities(state, bbox_data, pose_data)
    _add_entity_cells(state)
    _add_evidence_cells(state)
    _add_skeleton_edges(state)
    _add_semantic_edges(state)
    _add_spatial_edges(state)
    _add_evidence_edges(state)
    _add_audio_edges(state)
    _add_triad_edges(state)
    _add_hypercells(state)
    return _assemble(state, frame_meta)


def _normalize(coords_world: np.ndarray) -> np.ndarray:
    normed = (coords_world - OR_CENTER) / OR_HALF_SIZE
    return np.clip(normed, -1.0, 1.0).astype(np.float32)


def _ratio(distance_mm: float, scale_mm: float) -> float:
    return float(np.clip(distance_mm / scale_mm, 0.0, 1.0))


def _pair(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


def _posed_people(state: GraphBuildState) -> list[int]:
    return [eid for eid, info in state.entities.items() if info["is_human"] and info["pose"] is not None]


def _triads(state: GraphBuildState):
    """Present (actor, tool, target) entity triples of every functional triad."""
    for groups in FUNCTIONAL_TRIADS:
        yield from product(*([eid for eid in group if eid in state.entity_ids_present] for group in groups))


def _representative_node(state: GraphBuildState, eid: int) -> int:
    node_ids = state.entity_to_node_ids[eid]
    if len(node_ids) == NUM_JOINTS:
        if node_ids[PELVIS_IDX] != -1:
            return node_ids[PELVIS_IDX]
        return next((nid for nid in node_ids if nid != -1), -1)
    return node_ids[0]


def _anchor_nodes(state: GraphBuildState, eid: int) -> list[int]:
    """A person's visible wrists, else the entity's representative node."""
    if state.entities[eid]["is_human"]:
        node_ids = state.entity_to_node_ids[eid]
        wrists = [node_ids[idx] for idx in WRIST_GRAPH_INDICES if node_ids[idx] != -1]
        if wrists:
            return wrists
    rep = _representative_node(state, eid)
    return [rep] if rep != -1 else []


def _closest_pair(state: GraphBuildState, nodes_a: list[int], nodes_b: list[int]):
    best = None
    for na, nb in product(nodes_a, nodes_b):
        dist_mm = float(np.linalg.norm(state.rank0_world[nb] - state.rank0_world[na]))
        if best is None or dist_mm < best[0]:
            best = (dist_mm, na, nb)
    return best


def _offset(state: GraphBuildState, a: int, b: int) -> np.ndarray:
    return state.rank0_features[b][:3] - state.rank0_features[a][:3]


def _add_node(state: GraphBuildState, world: np.ndarray, subtype: float, eid: int, node_type: int) -> int:
    coords_norm = _normalize(world.astype(np.float32))
    state.rank0_features.append(np.array([*coords_norm, subtype, float(eid), float(node_type)], dtype=np.float32))
    state.rank0_world.append(world)
    return len(state.rank0_features) - 1


def _add_edge(state: GraphBuildState, a: int, b: int, rel, ratio: float, edge_type: int) -> int:
    """Append a rank-1 cell incident to rank-0 cells ``a`` and ``b``; return its cell index."""
    cell = state.n0 + state.n1
    state.rank1_features.append(np.array([*rel, ratio, float(edge_type)], dtype=np.float32))
    state.inc_0_1_src.extend([a, b])
    state.inc_0_1_tgt.extend([cell, cell])
    state.n1 += 1
    return cell


def _keep_best(state: GraphBuildState, pair: tuple[int, int], cell: int, ratio: float) -> None:
    best = state.inter_entity_best_edge.get(pair)
    if best is None or ratio < best[1]:
        state.inter_entity_best_edge[pair] = (cell, ratio)


def _add_hypercell(state: GraphBuildState, feature: list[float], edge_cells: list[int]) -> None:
    cell = state.n0 + state.n1 + state.n2
    state.rank2_features.append(np.array(feature, dtype=np.float32))
    state.inc_1_2_src.extend(edge_cells)
    state.inc_1_2_tgt.extend([cell] * len(edge_cells))
    state.n2 += 1


def _collect_entities(state: GraphBuildState, bbox_data: dict, pose_data: list[dict]) -> None:
    for obj_name, obj_info in bbox_data.items():
        if obj_name in object_name_to_id:
            eid = object_name_to_id[obj_name]
            state.entities[eid] = {
                "center": np.asarray(obj_info["center"], dtype=np.float64),
                "dimensions": np.asarray(obj_info["dimensions"], dtype=np.float64),
                "quaternion": np.asarray(obj_info["quaternion"], dtype=np.float64),
                "is_human": eid in HUMAN_ENTITIES,
                "pose": None,
            }
    for pose_info in pose_data:
        label = pose_info["label"]
        if label not in object_name_to_id:
            continue
        eid = object_name_to_id[label]
        kps = pose_info["keypoints"]
        if eid in state.entities:
            state.entities[eid]["pose"] = kps
        elif eid in HUMAN_ENTITIES:
            state.entities[eid] = {
                "center": kps[UPPER_BODY_INDICES[PELVIS_IDX], :3].astype(np.float64),
                "dimensions": np.array([500, 500, 1800], dtype=np.float64),
                "quaternion": np.array([0, 0, 0, 1], dtype=np.float64),
                "is_human": True,
                "pose": kps,
            }
    state.entity_ids_present = set(state.entities.keys())


def _add_entity_cells(state: GraphBuildState) -> None:
    """One cell per confident joint of a posed person, and one per other entity."""
    for eid, einfo in state.entities.items():
        if einfo["is_human"] and einfo["pose"] is not None:
            kps = einfo["pose"]
            state.entity_to_node_ids[eid] = [
                -1
                if float(kps[raw_idx, 3]) < CONF_THRESHOLD
                else _add_node(state, kps[raw_idx, :3].astype(np.float64), float(joint), eid, NODE_TYPE_JOINT)
                for joint, raw_idx in enumerate(UPPER_BODY_INDICES)
            ]
        else:
            state.entity_to_node_ids[eid] = [_add_node(state, einfo["center"].astype(np.float64), 0.0, eid, NODE_TYPE_OBJECT)]


def _add_evidence_cells(state: GraphBuildState) -> None:
    """Evidence cells: a visual cell per joint, screen cells for the robot, one audio cell."""
    for idx in range(len(state.rank0_features)):
        feat = state.rank0_features[idx]
        node_type, eid = int(feat[R0_COL_NODE_TYPE]), int(feat[R0_COL_ENTITY_ID])
        if node_type == NODE_TYPE_JOINT:
            subtypes = (EVIDENCE_SUBTYPE_VISUAL,)
        elif node_type == NODE_TYPE_OBJECT and eid == ENTITY_MAKO_ROBOT:
            subtypes = ROBOT_EVIDENCE_SUBTYPES
        else:
            continue
        for subtype in subtypes:
            ev_feat = feat.copy()
            ev_feat[R0_COL_NODE_TYPE] = float(NODE_TYPE_EVIDENCE)
            ev_feat[R0_COL_JOINT_TYPE] = float(subtype)
            state.physical_to_evidence_list[idx].append(len(state.rank0_features))
            state.rank0_features.append(ev_feat)
            state.rank0_world.append(state.rank0_world[idx])
    state.audio_node_idx = len(state.rank0_features)
    state.rank0_features.append(
        np.array([0.0, 0.0, 0.0, float(EVIDENCE_SUBTYPE_AUDIO), -1.0, float(NODE_TYPE_EVIDENCE)], dtype=np.float32)
    )
    state.rank0_world.append(OR_CENTER.astype(np.float64))
    state.n0 = len(state.rank0_features)


def _add_skeleton_edges(state: GraphBuildState) -> None:
    for eid in _posed_people(state):
        joint_ids = state.entity_to_node_ids[eid]
        for j1, j2 in SKELETON_EDGES:
            nid1, nid2 = joint_ids[j1], joint_ids[j2]
            if nid1 == -1 or nid2 == -1:
                continue
            dist_mm = float(np.linalg.norm(state.rank0_world[nid2] - state.rank0_world[nid1]))
            ratio = _ratio(dist_mm, SPATIAL_THRESHOLD_MM)
            cell = _add_edge(state, nid1, nid2, _offset(state, nid1, nid2), ratio, EDGE_TYPE_SKELETON)
            state.person_to_skel_edge_cells[eid].append(cell)


def _add_semantic_edges(state: GraphBuildState) -> None:
    """Connect the closest anchor nodes of every present semantic entity pair."""
    for eid_a, eid_b in SEMANTIC_EDGES:
        if eid_a not in state.entity_ids_present or eid_b not in state.entity_ids_present:
            continue
        nodes_a, nodes_b = _anchor_nodes(state, eid_a), _anchor_nodes(state, eid_b)
        if not nodes_a or not nodes_b:
            continue
        dist_mm, na, nb = _closest_pair(state, nodes_a, nodes_b)
        ratio = _ratio(dist_mm, SEMANTIC_THRESHOLD_MM)
        cell = _add_edge(state, na, nb, _offset(state, na, nb), ratio, EDGE_TYPE_SEMANTIC)
        state.inter_entity_best_edge[_pair(eid_a, eid_b)] = (cell, ratio)


def _add_spatial_edges(state: GraphBuildState) -> None:
    """Connect non-semantic entity pairs closer than ``SPATIAL_THRESHOLD_MM``."""
    for eid_a, eid_b in combinations(list(state.entity_ids_present), 2):
        if (eid_a, eid_b) in SEMANTIC_PAIRS:
            continue
        human_a, human_b = state.entities[eid_a]["is_human"], state.entities[eid_b]["is_human"]
        if human_a != human_b:
            _add_wrist_object_edges(state, *((eid_a, eid_b) if human_a else (eid_b, eid_a)))
        elif not human_a:
            _add_object_object_edge(state, eid_a, eid_b)
        else:
            _add_person_person_edge(state, eid_a, eid_b)


def _add_wrist_object_edges(state: GraphBuildState, human_eid: int, obj_eid: int) -> None:
    human, obj = state.entities[human_eid], state.entities[obj_eid]
    obj_node = _representative_node(state, obj_eid)
    if human["pose"] is None or obj_node == -1:
        return
    human_node_ids = state.entity_to_node_ids[human_eid]
    for graph_idx in WRIST_GRAPH_INDICES:
        wrist = human_node_ids[graph_idx]
        if wrist == -1:
            continue
        wrist_world = human["pose"][WRIST_RAW_MAP[graph_idx], :3].astype(np.float64)
        dist_mm = float(_point_to_obb_distance(wrist_world, obj["center"], obj["dimensions"], obj["quaternion"]))
        if dist_mm > SPATIAL_THRESHOLD_MM:
            continue
        ratio = _ratio(dist_mm, SPATIAL_THRESHOLD_MM)
        cell = _add_edge(state, wrist, obj_node, _offset(state, wrist, obj_node), ratio, EDGE_TYPE_SPATIAL)
        _keep_best(state, _pair(human_eid, obj_eid), cell, ratio)


def _add_object_object_edge(state: GraphBuildState, eid_a: int, eid_b: int) -> None:
    a, b = state.entities[eid_a], state.entities[eid_b]
    dist_mm = float(
        _obb_surface_distance(
            a["center"],
            a["dimensions"],
            a["quaternion"],
            b["center"],
            b["dimensions"],
            b["quaternion"],
            check_threshold=SPATIAL_THRESHOLD_MM,
        )
    )
    if dist_mm > SPATIAL_THRESHOLD_MM:
        return
    rep_a, rep_b = _representative_node(state, eid_a), _representative_node(state, eid_b)
    ratio = _ratio(dist_mm, SPATIAL_THRESHOLD_MM)
    cell = _add_edge(state, rep_a, rep_b, _offset(state, rep_a, rep_b), ratio, EDGE_TYPE_SPATIAL)
    _keep_best(state, _pair(eid_a, eid_b), cell, ratio)


def _add_person_person_edge(state: GraphBuildState, eid_a: int, eid_b: int) -> None:
    nodes_a, nodes_b = _anchor_nodes(state, eid_a), _anchor_nodes(state, eid_b)
    if not nodes_a or not nodes_b:
        return
    dist_mm, na, nb = _closest_pair(state, nodes_a, nodes_b)
    if dist_mm > SPATIAL_THRESHOLD_MM:
        return
    ratio = _ratio(dist_mm, SPATIAL_THRESHOLD_MM)
    cell = _add_edge(state, na, nb, _offset(state, na, nb), ratio, EDGE_TYPE_SPATIAL)
    _keep_best(state, _pair(eid_a, eid_b), cell, ratio)


def _add_evidence_edges(state: GraphBuildState) -> None:
    for physical, evidence_cells in state.physical_to_evidence_list.items():
        for ev_idx in evidence_cells:
            edge_type = EVIDENCE_EDGE_TYPES[int(state.rank0_features[ev_idx][R0_COL_JOINT_TYPE])]
            _add_edge(state, physical, ev_idx, np.zeros(3, dtype=np.float32), 0.0, edge_type)


def _add_audio_edges(state: GraphBuildState) -> None:
    """Broadcast the audio cell to the head joint of every posed person."""
    audio = state.audio_node_idx
    for eid in _posed_people(state):
        head = state.entity_to_node_ids[eid][HEAD_GRAPH_INDEX]
        if head == -1:
            continue
        cell = _add_edge(state, audio, head, _offset(state, audio, head), 0.0, EDGE_TYPE_AUDIO_BROADCAST)
        state.audio_edge_cells.append(cell)


def _add_triad_edges(state: GraphBuildState) -> None:
    """Close an actor-tool-target chain with an actor-target edge when that pair has none."""
    best = state.inter_entity_best_edge
    for actor, tool, target in _triads(state):
        if _pair(actor, tool) in best and _pair(tool, target) in best and _pair(actor, target) not in best:
            rep_a, rep_t = _representative_node(state, actor), _representative_node(state, target)
            if rep_a != -1 and rep_t != -1:
                dist_mm = float(np.linalg.norm(state.rank0_world[rep_t] - state.rank0_world[rep_a]))
                ratio = _ratio(dist_mm, SPATIAL_THRESHOLD_MM)
                cell = _add_edge(state, rep_a, rep_t, _offset(state, rep_a, rep_t), ratio, EDGE_TYPE_SPATIAL)
                best[_pair(actor, target)] = (cell, ratio)


def _add_hypercells(state: GraphBuildState) -> None:
    for eid in _posed_people(state):
        joint_ids = state.entity_to_node_ids[eid]
        root_idx = joint_ids[PELVIS_IDX]
        if root_idx != -1:
            root_pos = state.rank0_features[root_idx][:3]
        else:
            valid = [state.rank0_features[nid][:3] for nid in joint_ids if nid != -1]
            root_pos = np.mean(valid, axis=0) if valid else np.zeros(3, dtype=np.float32)
        feature = [*root_pos, float(eid), float(HYPER_TYPE_PERSON)]
        _add_hypercell(state, feature, state.person_to_skel_edge_cells.get(eid, []))
    best = state.inter_entity_best_edge
    for actor, tool, target in _triads(state):
        pairs = (_pair(actor, tool), _pair(tool, target), _pair(actor, target))
        if all(pair in best for pair in pairs):
            cells, ratios = zip(*(best[pair] for pair in pairs))
            _add_hypercell(state, [*ratios, -1.0, float(HYPER_TYPE_TRIAD)], list(cells))
    if state.audio_edge_cells:
        _add_hypercell(state, [0.0, 0.0, 0.0, -1.0, float(HYPER_TYPE_AUDIO)], state.audio_edge_cells)


def _assemble(state: GraphBuildState, frame_meta: dict) -> Data:
    x_0 = (
        torch.tensor(np.stack(state.rank0_features), dtype=torch.float32)
        if state.n0 > 0
        else torch.zeros((0, RANK0_FEAT_DIM))
    )
    x_1 = (
        torch.tensor(np.stack(state.rank1_features), dtype=torch.float32)
        if state.n1 > 0
        else torch.zeros((0, RANK1_FEAT_DIM))
    )
    x_2 = (
        torch.tensor(np.stack(state.rank2_features), dtype=torch.float32)
        if state.n2 > 0
        else torch.zeros((0, RANK2_FEAT_DIM))
    )
    hasse_src = state.inc_0_1_src + state.inc_0_1_tgt + state.inc_1_2_src + state.inc_1_2_tgt
    hasse_tgt = state.inc_0_1_tgt + state.inc_0_1_src + state.inc_1_2_tgt + state.inc_1_2_src
    edge_index_hasse = (
        torch.tensor([hasse_src, hasse_tgt], dtype=torch.long)
        if hasse_src
        else torch.zeros((2, 0), dtype=torch.long)
    )
    graph = Data()
    graph.x_0 = x_0
    graph.x_1 = x_1
    graph.x_2 = x_2
    graph.edge_index_hasse = edge_index_hasse
    graph.next_action = torch.tensor([frame_meta["next_action"]], dtype=torch.long)
    graph.robot_phase = torch.tensor([frame_meta["robot_phase"]], dtype=torch.long)
    raw_rels = frame_meta["relation_labels"]
    if raw_rels.numel() > 0:
        raw_rels = raw_rels.to(torch.long)
        present_tensor = torch.tensor(list(state.entity_ids_present), dtype=torch.long, device=raw_rels.device)
        sub_ok = torch.isin(raw_rels[:, 0], present_tensor)
        obj_ok = torch.isin(raw_rels[:, 2], present_tensor)
        filtered_rels = raw_rels[sub_ok & obj_ok]
        if filtered_rels.numel() > 0:
            s, p, o = (filtered_rels[:, 0], filtered_rels[:, 1], filtered_rels[:, 2])
            assert int(s.min()) >= 0 and int(s.max()) < len(object_name_to_id), "Subject ID out of bounds"
            assert int(o.min()) >= 0 and int(o.max()) < len(object_name_to_id), "Object ID out of bounds"
            assert int(p.min()) >= 0 and int(p.max()) < len(predicate_name_to_id), "Predicate ID out of bounds"
        graph.relation_labels = filtered_rels
    else:
        graph.relation_labels = torch.zeros((0, 3), dtype=torch.long)
    graph.num_relation_labels = torch.tensor([graph.relation_labels.size(0)], dtype=torch.long)
    graph.sequence = frame_meta["sequence"]
    graph.frame_idx = frame_meta["frame_idx"]
    num_cells = state.n0 + state.n1 + state.n2
    assert graph.x_0.size(0) == state.n0
    assert graph.x_1.size(0) == state.n1
    assert graph.x_2.size(0) == state.n2
    if graph.edge_index_hasse.numel() > 0:
        assert int(graph.edge_index_hasse.min()) >= 0
        assert int(graph.edge_index_hasse.max()) < num_cells
    return graph
