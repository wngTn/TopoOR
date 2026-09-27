import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.datasets.constants import (
    CAMERA_IMAGE_SIZE,
    EVIDENCE_SUBTYPE_AUDIO,
    EVIDENCE_SUBTYPE_JSON,
    EVIDENCE_SUBTYPE_SCREEN_IMAGE,
    EVIDENCE_SUBTYPE_TEXT,
    EVIDENCE_SUBTYPE_VISUAL,
    HYPER_TYPE_AUDIO,
    HYPER_TYPE_PERSON,
    HYPER_TYPE_TRIAD,
    NODE_TYPE_EVIDENCE,
    NODE_TYPE_JOINT,
    NODE_TYPE_OBJECT,
    NUM_JOINTS,
    OR_CENTER,
    OR_HALF_SIZE,
    R0_COL_ENTITY_ID,
    R0_COL_JOINT_TYPE,
    R0_COL_NODE_TYPE,
    R1_COL_EDGE_TYPE,
    R2_COL_HYPER_TYPE,
    R2_COL_PERSON_ENTITY_ID,
    object_name_to_id,
)
from src.models.ops import scatter_add, scatter_softmax
from src.utils.camera_functions import world_3d_to_img_2d

PERSON_ROLE_IDS = [
    object_name_to_id["assistant_surgeon"],
    object_name_to_id["head_surgeon"],
    object_name_to_id["nurse"],
    object_name_to_id["mps"],
    object_name_to_id["anaesthetist"],
    object_name_to_id["student"],
    object_name_to_id["unrelated_person"],
    object_name_to_id["circulator"],
]
PERSON_ROLE_ID_SET = set(PERSON_ROLE_IDS)
OBJECT_TYPE_IDS = sorted({eid for eid in object_name_to_id.values() if eid not in PERSON_ROLE_ID_SET})
NUM_OBJECT_TYPES = len(OBJECT_TYPE_IDS)

_MAX_ENTITY_ID = max(object_name_to_id.values()) + 1


def build_entity_lut(lookup: dict[int, int], unknown_idx: int) -> torch.Tensor:
    lut = torch.full((_MAX_ENTITY_ID,), unknown_idx, dtype=torch.long)
    for eid, index in lookup.items():
        lut[eid] = index
    return lut


class SymmetricEndpointSummary(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim * 3, out_dim)

    def forward(self, x_u: torch.Tensor, x_v: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([x_u + x_v, (x_u - x_v).abs(), x_u * x_v], dim=-1))


class TriadAttentionPool(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, attn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, attn_dim, bias=False)
        self.key_proj = nn.Linear(key_dim, attn_dim, bias=False)
        self.scale = attn_dim**-0.5
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self, query_features: torch.Tensor, edge_features: torch.Tensor, edge_to_triad: torch.Tensor, num_triads: int
    ) -> torch.Tensor:
        if edge_features.size(0) == 0:
            return edge_features.new_zeros((num_triads, edge_features.size(-1)))

        q = self.query_proj(query_features)
        q_per_edge = q[edge_to_triad]
        k = self.key_proj(edge_features)

        logits = (q_per_edge * k).sum(dim=-1) * self.scale
        alpha = scatter_softmax(logits, edge_to_triad)
        alpha = self.attn_drop(alpha)

        weighted = edge_features * alpha.unsqueeze(-1)
        return scatter_add(weighted, edge_to_triad, dim_size=num_triads)


class RoleConditionedPool(nn.Module):
    def __init__(self, dim: int, role_dim: int, dropout: float = 0.1):
        super().__init__()
        self.key = nn.Linear(dim, dim, bias=False)
        self.query_proj = nn.Linear(role_dim, dim, bias=False)
        self.scale = dim**-0.5
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor, role_emb: torch.Tensor, dim_size: int | None = None
    ) -> torch.Tensor:
        k = self.key(x)
        q = self.query_proj(role_emb)
        q_per_joint = q[batch]

        logits = (q_per_joint * k).sum(dim=-1) * self.scale
        alpha = scatter_softmax(logits, batch)
        alpha = self.attn_drop(alpha)

        out = x * alpha.unsqueeze(-1)
        return scatter_add(out, batch, dim_size=dim_size)


class Projection:
    def _denormalize_coords(self, norm_coords: torch.Tensor) -> torch.Tensor:
        center = torch.as_tensor(OR_CENTER, device=norm_coords.device)
        half = torch.as_tensor(OR_HALF_SIZE, device=norm_coords.device)
        return norm_coords * half + center

    def _apply_affine_2d(self, xy: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
        affine = affine.unsqueeze(1)
        xy_homo = F.pad(xy, (0, 1), mode="constant", value=1.0)
        return torch.einsum("...ij,...j->...i", affine, xy_homo)

    def _project_and_normalize(
        self,
        world_pos: torch.Tensor,
        cam_params: torch.Tensor,
        affine: torch.Tensor,
        orig_img_wh: tuple[int, int],
        input_img_wh: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        poses_xyz = world_pos.unsqueeze(1).unsqueeze(1)
        xy_img, xy_valid = world_3d_to_img_2d(poses_xyz, cam_params)
        xy_img = xy_img.squeeze(2)
        xy_valid = xy_valid.squeeze(2)
        W_orig, H_orig = orig_img_wh
        inside_w = (xy_img[..., 0] >= 0) & (xy_img[..., 0] < W_orig)
        inside_h = (xy_img[..., 1] >= 0) & (xy_img[..., 1] < H_orig)
        valid = inside_w & inside_h & xy_valid.squeeze(-1)
        xy_clamped = xy_img.clamp(min=0.0)
        xy_clamped[..., 0] = xy_clamped[..., 0].clamp(max=W_orig - 1.0)
        xy_clamped[..., 1] = xy_clamped[..., 1].clamp(max=H_orig - 1.0)
        xy_clamped = xy_clamped.unsqueeze(3)
        affine_expanded = affine[:, None, None, :, :]
        xy_input = self._apply_affine_2d(xy_clamped, affine_expanded)
        W_inp, H_inp = input_img_wh
        xy_norm = xy_input.clone()
        xy_norm[..., 0] = xy_norm[..., 0] / W_inp
        xy_norm[..., 1] = xy_norm[..., 1] / H_inp
        valid = valid.unsqueeze(-1)
        return (xy_norm, valid)

class EvidenceEncoding(Projection):
    def _finish_evidence(self, features, dtype):
        return features

    def _evidence_streams(self):
        yield EVIDENCE_SUBTYPE_TEXT, "screen_txt", self.screen_proj_txt
        yield EVIDENCE_SUBTYPE_JSON, "screen_json", self.screen_proj_json
        yield EVIDENCE_SUBTYPE_SCREEN_IMAGE, "screen_image", self.screen_backbone
        yield EVIDENCE_SUBTYPE_AUDIO, "audio", self.audio_proj

    def _process_evidence_nodes(self, bg, encoded):
        evidence = (bg.rank == 0) & (bg.x[:, R0_COL_NODE_TYPE].long() == NODE_TYPE_EVIDENCE)
        if not evidence.any():
            return encoded
        frame_data = next(
            (value for name in ("images", "screen_txt", "audio") if (value := getattr(bg, name, None)) is not None),
            None,
        )
        batch = bg.batch.long()
        batch_size = int(batch.max().item() + 1) if batch.numel() else 0
        if frame_data is None or batch_size == 0:
            return encoded
        frame_count = frame_data.size(0)
        assert frame_count % batch_size == 0, "Frame count must be divisible by batch size"
        window = frame_count // batch_size
        time = bg.t.long()
        assert (time >= 0).all() and (time < window).all(), f"bg.t out of range [0,{window})"
        frame_index = batch * window + time
        subtype = bg.x[:, R0_COL_JOINT_TYPE].long()
        visual = evidence & (subtype == EVIDENCE_SUBTYPE_VISUAL)
        calibrated = all(getattr(bg, name, None) is not None for name in ("images", "cam_params", "affine"))
        if visual.any() and calibrated:
            self._encode_visual(bg, encoded, frame_index, torch.nonzero(visual).flatten())
        for kind, attribute, encoder in self._evidence_streams():
            indices = torch.nonzero(evidence & (subtype == kind)).flatten()
            if indices.numel():
                inputs = getattr(bg, attribute)
                encoded[indices] = self._finish_evidence(encoder(inputs[frame_index[indices]]), encoded.dtype)
        return encoded

    def _encode_visual(self, bg, encoded, frame_index, indices):
        frames, views, _, height, width = bg.images.shape
        feature_maps = list(self.visual_backbone(bg.images).values())
        positions = self._denormalize_coords(bg.x[indices, :3])
        image_index = frame_index[indices]
        coordinates, valid = self._project_and_normalize(
            positions, bg.cam_params[image_index], bg.affine[image_index], CAMERA_IMAGE_SIZE, (width, height)
        )
        counts = torch.bincount(image_index, minlength=frames)
        max_nodes = int(counts.max().item()) if counts.numel() else 0
        if max_nodes == 0:
            return
        query = encoded.new_zeros((frames, max_nodes, self.cell_dim))
        poses = encoded.new_zeros((frames, views, max_nodes, 1, 2))
        visible = encoded.new_zeros((frames, views, max_nodes, 1), dtype=torch.bool)
        order = torch.argsort(image_index, stable=True)
        rows = image_index[order]
        starts = encoded.new_zeros(frames + 1, dtype=torch.long)
        starts[1:] = torch.cumsum(counts, dim=0)
        columns = torch.arange(len(image_index), device=encoded.device) - starts[rows]
        sources = indices[order]
        query[rows, columns] = encoded[sources]
        poses[rows, :, columns] = coordinates[order].squeeze(2).to(poses.dtype)
        visible[rows, :, columns] = valid[order].squeeze(2)
        projected = self.proj_attn(query=query, poses_xy_norm=poses, feature_maps=feature_maps)
        pooled = self._pool_views(projected.squeeze(3), visible.squeeze(3))
        encoded[sources] = self._finish_evidence(pooled[rows, columns], encoded.dtype)

    def _pool_views(self, features, valid):
        logits = self.view_gate(features).squeeze(-1)
        all_invalid = ~valid.any(dim=1, keepdim=True)
        logits = logits.masked_fill(~valid, float("-inf")).masked_fill(all_invalid, 0.0)
        attention = torch.softmax(logits, dim=1).masked_fill(~valid, 0.0)
        features = features.masked_fill(~valid.unsqueeze(-1), 0.0)
        return (attention.unsqueeze(-1) * features).sum(dim=1)

def incident_features(bg, features, source_rank, targets):
    source, destination = bg.edge_index
    edges = bg.edge_index[:, (bg.rank[source] == source_rank) & (bg.rank[destination] == 2)]
    remap = torch.full((bg.x.size(0),), -1, device=features.device, dtype=torch.long)
    remap[targets] = torch.arange(targets.size(0), device=features.device)
    compact = remap[edges[1]]
    valid = compact >= 0
    return features[edges[0, valid]], compact[valid]


def edge_endpoints(bg):
    targets = torch.nonzero(bg.rank == 1, as_tuple=False).squeeze(-1)
    source, destination = bg.edge_index
    edges = bg.edge_index[:, (bg.rank[source] == 0) & (bg.rank[destination] == 1)]
    assert edges.size(1) == 2 * targets.numel(), "Each edge must have exactly two vertex incidences"
    order = torch.argsort(edges[1], stable=True)
    destination, source = edges[1, order], edges[0, order]
    assert torch.all(destination[0::2] == destination[1::2]), "Edge incidences are not paired"
    unique = destination[0::2]
    positions = torch.searchsorted(unique.contiguous(), targets)
    assert torch.all(unique[positions] == targets), "An edge is missing its vertex incidences"
    return source[0::2][positions], source[1::2][positions]


class CellEncoding(EvidenceEncoding):
    def _encode_rank0(self, bg, cells):
        count = cells.size(0)
        if count == 0:
            return cells.new_zeros((0, self.cell_dim))
        joint_type = cells[:, R0_COL_JOINT_TYPE].long()
        entity = cells[:, R0_COL_ENTITY_ID].long()
        node_type = cells[:, R0_COL_NODE_TYPE].long()
        position = self.r0_pos_proj(cells[:, :3])
        identity = cells.new_zeros((count, self.r0_id_dim))
        type_index = torch.full((count,), 3, dtype=torch.long, device=cells.device)
        joints = torch.nonzero(node_type == NODE_TYPE_JOINT).flatten()
        if joints.numel():
            role = self.person_role_emb(self.role_lut[entity[joints].clamp(0, _MAX_ENTITY_ID - 1)])
            joint = self.joint_type_emb(joint_type[joints].clamp(0, NUM_JOINTS - 1))
            identity[joints] = torch.cat([role, joint], dim=-1)
            type_index[joints] = 0
        objects = torch.nonzero(node_type == NODE_TYPE_OBJECT).flatten()
        if objects.numel():
            identity[objects] = self.object_type_emb(self.object_lut[entity[objects].clamp(0, _MAX_ENTITY_ID - 1)])
            type_index[objects] = 1
        evidence = torch.nonzero(node_type == NODE_TYPE_EVIDENCE).flatten()
        if evidence.numel():
            identity[evidence] = self.evidence_subtype_emb(joint_type[evidence].clamp(0, 4))
            type_index[evidence] = 2
        return torch.cat([position, identity], dim=-1) + self.node_type_emb(type_index)

    def _encode_rank1(self, bg, cells, features):
        if cells.size(0) == 0:
            return cells.new_zeros((0, self.cell_dim))
        edge_type = cells[:, R1_COL_EDGE_TYPE].long()
        assert int(edge_type.max()) < self.edge_type_emb.num_embeddings, "Edge type exceeds its embedding table"
        edge_type = edge_type.clamp(0, self.edge_type_emb.num_embeddings - 1)
        source, destination = edge_endpoints(bg)
        geometry = self.r1_geo_proj(cells[:, :4])
        identity = self.edge_type_emb(edge_type)
        endpoints = self.r1_endpoint_summary(features[source], features[destination])
        return self.r1_fuse(geometry, identity, endpoints)

    def _encode_person(self, bg, cells, features, targets):
        position = self.r2_person_pos_proj(cells[:, :3].contiguous())
        entities = cells[:, R2_COL_PERSON_ENTITY_ID].long().clamp(0, _MAX_ENTITY_ID - 1)
        role = self.person_role_emb(self.role_lut[entities])
        identity = self.r2_person_role_proj(role)
        joints, destination = incident_features(bg, features, 0, targets)
        if joints.size(0):
            pooled = self.r2_person_joint_pool(joints, destination, role_emb=role, dim_size=targets.size(0))
        else:
            pooled = cells.new_zeros((targets.size(0), self.cell_dim))
        joints = self.r2_person_joint_proj(pooled)
        return self.r2_person_proj(torch.cat([identity, position, joints], dim=-1))

    def _encode_triad(self, bg, cells, features, targets):
        distances = self.r2_triad_dist_proj(cells[:, :3].contiguous())
        edges, destination = incident_features(bg, features, 1, targets)
        pooled = self.r2_triad_attn_pool(distances, edges, destination, targets.size(0))
        return torch.cat([distances, self.r2_triad_edge_proj(pooled)], dim=-1)

    def _encode_rank2(self, bg, cells, features):
        output = cells.new_zeros((cells.size(0), self.cell_dim))
        if cells.size(0) == 0:
            return output
        kind = cells[:, R2_COL_HYPER_TYPE].long()
        targets = torch.nonzero(bg.rank == 2).flatten()
        persons, triads = kind == HYPER_TYPE_PERSON, kind == HYPER_TYPE_TRIAD
        if persons.any():
            output[persons] = self._encode_person(bg, cells[persons], features, targets[persons])
        if triads.any():
            output[triads] = self._encode_triad(bg, cells[triads], features, targets[triads])
        return output

    def _encode_cells(self, bg):
        features = bg.x.new_zeros((bg.x.size(0), self.cell_dim))
        vertices, edges, groups = bg.rank == 0, bg.rank == 1, bg.rank == 2
        if vertices.any():
            features[vertices] = self._encode_rank0(bg, bg.x[vertices])
        features = self._process_evidence_nodes(bg, features)
        evidence = vertices & (bg.x[:, R0_COL_NODE_TYPE].long() == NODE_TYPE_EVIDENCE)
        if evidence.any():
            indices = torch.nonzero(evidence).flatten()
            types = torch.full((indices.size(0),), 2, dtype=torch.long, device=features.device)
            features[indices] = features[indices] + self.node_type_emb(types)
        if vertices.any():
            features[vertices] = features[vertices] + self.r0_refine(features[vertices])
        if edges.any():
            features[edges] = self._encode_rank1(bg, bg.x[edges], features)
        if groups.any():
            features[groups] = self._encode_rank2(bg, bg.x[groups], features)
        return self.cell_ln(features)

class WorkflowRecognitionEncoding(CellEncoding):
    def _encode_rank0(self, bg, cells):
        features = super()._encode_rank0(bg, cells)
        return self.r0_drop(features) if cells.size(0) else features

    def _encode_person(self, bg, cells, features, targets):
        return self.r2_drop(super()._encode_person(bg, cells, features, targets))

    def _encode_triad(self, bg, cells, features, targets):
        return self.r2_drop(super()._encode_triad(bg, cells, features, targets))

    def _encode_rank2(self, bg, cells, features):
        encoded = super()._encode_rank2(bg, cells, features)
        audio = cells[:, R2_COL_HYPER_TYPE].long() == HYPER_TYPE_AUDIO
        if audio.any():
            targets = torch.nonzero(bg.rank == 2).flatten()[audio]
            encoded[audio] = self._encode_triad(bg, cells[audio], features, targets)
        return encoded

    def _finish_evidence(self, features, dtype):
        return self.evidence_drop(features).to(dtype)


RelationPredictionEncoding = CellEncoding


class CellInitialization:
    def _init_rank0(self, cfg):
        self.r0_pos_dim = cfg.cell_dim // 2
        self.r0_id_dim = cfg.cell_dim - self.r0_pos_dim
        self.r0_role_dim = self.r0_id_dim * 2 // 3
        self.r0_joint_dim = self.r0_id_dim - self.r0_role_dim
        self.r0_pos_proj = nn.Sequential(nn.Linear(3, self.r0_pos_dim), nn.GELU(), nn.LayerNorm(self.r0_pos_dim))
        self.person_role_emb = nn.Embedding(len(PERSON_ROLE_IDS) + 1, self.r0_role_dim)
        self.joint_type_emb = nn.Embedding(NUM_JOINTS + 1, self.r0_joint_dim)
        self.object_type_emb = nn.Embedding(NUM_OBJECT_TYPES + 1, self.r0_id_dim)
        self.evidence_subtype_emb = nn.Embedding(5 + 1, self.r0_id_dim)
        self.node_type_emb = nn.Embedding(4, cfg.cell_dim)
        person_role_lookup = {eid: idx for idx, eid in enumerate(PERSON_ROLE_IDS)}
        object_type_lookup = {eid: idx for idx, eid in enumerate(OBJECT_TYPE_IDS)}
        self.register_buffer("role_lut", build_entity_lut(person_role_lookup, unknown_idx=len(PERSON_ROLE_IDS)))
        self.register_buffer("object_lut", build_entity_lut(object_type_lookup, unknown_idx=NUM_OBJECT_TYPES))
        self.r0_refine = nn.Sequential(nn.LayerNorm(cfg.cell_dim), nn.Linear(cfg.cell_dim, cfg.cell_dim), nn.GELU())

    def _init_person(self, cfg):
        self.r2p_ent_dim = cfg.cell_dim // 4
        self.r2p_pos_dim = cfg.cell_dim // 8
        self.r2p_joint_pool_dim = cfg.cell_dim - self.r2p_ent_dim - self.r2p_pos_dim
        self.r2_person_pos_proj = nn.Sequential(
            nn.Linear(3, self.r2p_pos_dim), nn.GELU(), nn.LayerNorm(self.r2p_pos_dim)
        )
        self.r2_person_role_proj = nn.Linear(self.r0_role_dim, self.r2p_ent_dim)
        self.r2_person_joint_pool = RoleConditionedPool(
            dim=cfg.cell_dim, role_dim=self.r0_role_dim, dropout=cfg.pool_dropout
        )
        self.r2_person_joint_proj = nn.Sequential(
            nn.LayerNorm(cfg.cell_dim), nn.Linear(cfg.cell_dim, self.r2p_joint_pool_dim), nn.GELU()
        )
        self.r2_person_proj = nn.Sequential(
            nn.Linear(cfg.cell_dim, cfg.cell_dim), nn.GELU(), nn.LayerNorm(cfg.cell_dim)
        )

    def _init_triad(self, cfg):
        self.r2t_dist_dim = cfg.cell_dim // 2
        self.r2t_edge_pool_dim = cfg.cell_dim - self.r2t_dist_dim
        self.r2_triad_dist_proj = nn.Sequential(
            nn.Linear(3, self.r2t_dist_dim), nn.GELU(), nn.LayerNorm(self.r2t_dist_dim)
        )
        self.r2_triad_attn_pool = TriadAttentionPool(
            query_dim=self.r2t_dist_dim, key_dim=cfg.cell_dim, attn_dim=cfg.cell_dim // 2, dropout=cfg.pool_dropout
        )
        self.r2_triad_edge_proj = nn.Sequential(
            nn.LayerNorm(cfg.cell_dim), nn.Linear(cfg.cell_dim, self.r2t_edge_pool_dim), nn.GELU()
        )
