from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.init import constant_, xavier_uniform_


def _num_segments(index: torch.Tensor, dim_size: int | None) -> int:
    if dim_size is not None:
        return int(dim_size)
    return int(index.max().item()) + 1 if index.numel() else 0


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int | None = None) -> torch.Tensor:
    """Sum the rows of ``src`` into the segments ``index``."""
    out = src.new_zeros((_num_segments(index, dim_size),) + tuple(src.shape[1:]))
    if src.numel() == 0:
        return out
    if src.dim() > 1:
        index = index.reshape((-1,) + (1,) * (src.dim() - 1)).expand_as(src)
    return out.scatter_add_(0, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int | None = None) -> torch.Tensor:
    """Average the rows of ``src`` over the segments ``index``; empty segments are zero."""
    n = _num_segments(index, dim_size)
    summed = scatter_add(src, index, n)
    cnt = src.new_zeros(n).scatter_add_(0, index, src.new_ones(index.shape[0]))
    cnt = cnt.clamp_min(1.0).reshape((-1,) + (1,) * (src.dim() - 1))
    return summed / cnt


def scatter_softmax(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Softmax of the finite 1-D logits ``src`` within each segment ``index``, shifted by the segment max."""
    if src.numel() == 0:
        return src
    n = _num_segments(index, None)
    seg_max = src.new_zeros(n).scatter_reduce_(0, index, src.detach(), "amax", include_self=False)
    e = (src - seg_max.index_select(0, index)).exp()
    denom = src.new_zeros(n).scatter_add_(0, index, e)
    return e / (denom.index_select(0, index) + 1e-16)


def det_grid_sample(inp: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Use native CPU sampling; CUDA needs gather for deterministic backward."""
    if not inp.is_cuda:
        return F.grid_sample(inp, grid, align_corners=False)
    return _gather_grid_sample(inp, grid)


def _gather_grid_sample(inp, grid):
    N, C, H, W = inp.shape
    _, Ho, Wo, _ = grid.shape
    gx, gy = grid[..., 0], grid[..., 1]
    # align_corners=False: map [-1,1] to pixel-center coords
    ix = ((gx + 1.0) * W - 1.0) / 2.0
    iy = ((gy + 1.0) * H - 1.0) / 2.0
    ix0 = torch.floor(ix)
    iy0 = torch.floor(iy)
    ix1, iy1 = ix0 + 1.0, iy0 + 1.0
    wx1, wy1 = ix - ix0, iy - iy0
    wx0, wy0 = 1.0 - wx1, 1.0 - wy1
    inp_flat = inp.reshape(N, C, H * W)

    def corner(ixc, iyc, wgt):
        valid = (ixc >= 0) & (ixc <= W - 1) & (iyc >= 0) & (iyc <= H - 1)
        flat = iyc.clamp(0, H - 1).long() * W + ixc.clamp(0, W - 1).long()  # [N, Ho, Wo]
        flat = flat.reshape(N, 1, Ho * Wo).expand(N, C, Ho * Wo)
        vals = torch.gather(inp_flat, 2, flat).reshape(N, C, Ho, Wo)
        return vals * (wgt * valid).unsqueeze(1)

    return (
        corner(ix0, iy0, wx0 * wy0)
        + corner(ix1, iy0, wx1 * wy0)
        + corner(ix0, iy1, wx0 * wy1)
        + corner(ix1, iy1, wx1 * wy1)
    )


class ProjAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_points: int,
        input_dims: tuple[int, ...] = (128, 256, 512),
    ):
        """
        Projective Attention Module
        :param input_dims: List of channel counts for the feature maps.
                           If provided, 1x1 convolutions will project them to d_model.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_levels = len(input_dims)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.input_projs = nn.ModuleList([nn.Conv2d(in_dim, d_model, kernel_size=1) for in_dim in input_dims])
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.rayconv = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.rayconv.weight)
        constant_(self.rayconv.bias, 0.0)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias, 0.0)
        for proj in self.input_projs:
            xavier_uniform_(proj.weight)
            if proj.bias is not None:
                constant_(proj.bias, 0.0)

    def forward(self, query: torch.Tensor, poses_xy_norm: torch.Tensor, feature_maps: list[torch.Tensor]):
        projected_maps = []
        if len(feature_maps) != len(self.input_projs):
            raise ValueError(f"Expected {len(self.input_projs)} feature levels, got {len(feature_maps)}")
        for i, fm in enumerate(feature_maps):
            B, V, C_in, H, W = fm.shape
            fm_reshaped = fm.view(B * V, C_in, H, W)
            fm_proj = self.input_projs[i](fm_reshaped)
            fm_proj = fm_proj.view(B, V, self.d_model, H, W)
            projected_maps.append(fm_proj)
        feature_maps = projected_maps
        nfeat_level = len(feature_maps)
        shapes = [feature.shape[-2:] for feature in feature_maps]
        spatial_shapes = torch.tensor(shapes, device=feature_maps[0].device, dtype=torch.long)
        B, V, N, J, D_pose = poses_xy_norm.shape
        NJ = N * J
        B_eff = B * V
        poses_xy_01 = poses_xy_norm[..., :2]
        poses_xy_01 = poses_xy_01.view(B_eff, NJ, 1, 2)
        poses_xy_01_levels = poses_xy_01.expand(B_eff, NJ, nfeat_level, 2)
        grid_coords = torch.clamp(poses_xy_01_levels * 2.0 - 1.0, -1.1, 1.1)
        reference_point_features = []
        for lvl in range(nfeat_level):
            fm = feature_maps[lvl]
            B_f, V_f, C_l, H_l, W_l = fm.shape
            fm_bv = fm.view(B_eff, C_l, H_l, W_l)
            grid_l = grid_coords[:, :, lvl : lvl + 1, :]
            feats = det_grid_sample(fm_bv, grid_l)
            feats = feats.squeeze(-1).permute(0, 2, 1)
            reference_point_features.append(feats)
        reference_point_features = torch.stack(reference_point_features, dim=2)
        feature_maps_bv = [fm.view(B_eff, fm.shape[2], fm.shape[3], fm.shape[4]) for fm in feature_maps]
        feature_maps_f = torch.cat([x.flatten(2) for x in feature_maps_bv], dim=-1).permute(0, 2, 1)
        value = self.rayconv(feature_maps_f)
        value = value.view(B_eff, -1, self.n_heads, self.d_model // self.n_heads)
        query_bv = repeat(query, "B NJ D -> (B V) NJ D", V=V)
        fused_q = reference_point_features + query_bv.unsqueeze(2)
        sampling_offsets_px = self.sampling_offsets(fused_q)
        sampling_offsets_px = sampling_offsets_px.view(B_eff, NJ, nfeat_level, self.n_heads, self.n_points, 2)
        sampling_offsets_px = sampling_offsets_px.permute(0, 1, 3, 2, 4, 5)
        attention_weights = self.attention_weights(fused_q)
        attention_weights = rearrange(
            attention_weights, "b nj l (h p) -> b nj h (l p)", h=self.n_heads, p=self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = rearrange(attention_weights, "b nj h (l p) -> b nj h l p", l=nfeat_level, p=self.n_points)
        poses_xy_01_levels = poses_xy_01_levels.unsqueeze(2).unsqueeze(-2)
        offset_norm = spatial_shapes.flip(-1).to(sampling_offsets_px).view(1, 1, 1, nfeat_level, 1, 2)
        sampling_offset = sampling_offsets_px / offset_norm
        sampling_locations = poses_xy_01_levels + sampling_offset
        with torch.autocast(device_type=value.device.type, enabled=False):
            output = deform_core_pytorch(
                value.float(), shapes, sampling_locations.contiguous().float(), attention_weights.float()
            )
        output = self.output_proj(output)
        C_out = output.shape[-1]
        output = output.view(B, V, N, J, C_out)
        return output


def deform_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_list.append(det_grid_sample(value_l_, sampling_grid_l_))
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()
