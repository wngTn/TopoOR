from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax

from src.models.ops import scatter_add

DROP_PATH_RATE = 0.1
RANK_BIAS_DIM = 16
FFN_MULT = 2


def _add_global_tokens(
    x: torch.Tensor, rank: torch.Tensor, t: torch.Tensor, batch: torch.Tensor, token_rank_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append one [CLS]-style global token per graph."""
    device = x.device
    B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    N = x.size(0)
    x_tok = torch.zeros(B, x.size(1), device=device, dtype=x.dtype)
    rank_tok = torch.full((B,), token_rank_id, device=device, dtype=rank.dtype)
    batch_tok = torch.arange(B, device=device, dtype=batch.dtype)
    t_tok = t.new_zeros(B).scatter_reduce_(0, batch, t, "amax", include_self=False)
    orig_mask = torch.ones(N + B, dtype=torch.bool, device=device)
    orig_mask[N:] = False
    return (
        torch.cat([x, x_tok]),
        torch.cat([rank, rank_tok]),
        torch.cat([t, t_tok]),
        torch.cat([batch, batch_tok]),
        orig_mask,
    )


def _global_token_edges(batch: torch.Tensor, is_token: torch.Tensor) -> torch.Tensor:
    """Bidirectional edges between global tokens and all real nodes in same graph."""
    device = batch.device
    tok_idx = torch.nonzero(is_token, as_tuple=False).view(-1)
    node_idx = torch.nonzero(~is_token, as_tuple=False).view(-1)
    if tok_idx.numel() == 0 or node_idx.numel() == 0:
        return torch.zeros(2, 0, device=device, dtype=torch.long)
    B = int(batch.max().item()) + 1
    tok_by_graph = torch.full((B,), -1, device=device, dtype=torch.long)
    tok_by_graph[batch[tok_idx]] = tok_idx
    node_toks = tok_by_graph[batch[node_idx]]
    valid = node_toks >= 0
    src, tgt = (node_idx[valid], node_toks[valid])
    return torch.stack([torch.cat([src, tgt]), torch.cat([tgt, src])])


class HigherOrderAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, num_rank_types: int, dropout: float):
        super().__init__()
        assert hidden_dim % heads == 0
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.rank_bias_src = nn.Embedding(num_rank_types, RANK_BIAS_DIM)
        self.rank_bias_tgt = nn.Embedding(num_rank_types, RANK_BIAS_DIM)
        self.rank_bias_proj = nn.Linear(RANK_BIAS_DIM, heads, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, rank):
        N = h.size(0)
        qkv = self.qkv(h).reshape(N, 3, self.heads, self.head_dim)
        q, k, v = (qkv[:, 0], qkv[:, 1], qkv[:, 2])
        src, tgt = (edge_index[0], edge_index[1])
        q_e = q[tgt]
        k_e = k[src]
        v_e = v[src]
        scores = (q_e * k_e).sum(-1) * self.scale
        src_rank_emb = self.rank_bias_src(rank[src])
        tgt_rank_emb = self.rank_bias_tgt(rank[tgt])
        pair_emb = src_rank_emb * tgt_rank_emb
        pair_bias = self.rank_bias_proj(pair_emb)
        scores = scores + pair_bias
        attn = pyg_softmax(scores, index=tgt, num_nodes=N)
        attn = self.attn_drop(attn)
        msg = attn.unsqueeze(-1) * v_e
        out = scatter_add(msg, tgt, dim_size=N)
        return self.out_drop(self.out_proj(out.reshape(N, -1)))


class HigherOrderAttentionLayer(nn.Module):
    """Pre-norm transformer block with rank-aware attention and per-graph stochastic depth."""

    def __init__(self, hidden_dim: int, heads: int, num_rank_types: int, dropout: float, drop_path: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = HigherOrderAttention(hidden_dim, heads, num_rank_types, dropout)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, FFN_MULT * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(FFN_MULT * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.drop_path_rate = drop_path

    def _drop_path(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Per-graph stochastic depth during training.

        Each graph in the batch is independently kept (with rescaling) or
        dropped, so co-adaptation is broken without coupling graphs.
        """
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep = 1.0 - self.drop_path_rate
        B = int(batch.max().item()) + 1 if batch.numel() else 1
        mask_g = (torch.rand(B, device=x.device) < keep).float() / keep
        return x * mask_g[batch].unsqueeze(-1)

    def forward(self, h, edge_index, rank, batch):
        h = h + self._drop_path(self.attn(self.norm_attn(h), edge_index, rank), batch)
        h = h + self._drop_path(self.ffn(self.norm_ffn(h)), batch)
        return h


class CCMessagePassingTransformer(nn.Module):
    """Rank-aware transformer over complex cells, with one global token per graph and self loops."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int,
        heads: int,
        dropout: float,
        max_rank: int,
        max_time_steps: int,
        embed_dropout: float,
    ):
        super().__init__()
        assert hidden_features % heads == 0
        self.out_features = out_features
        self.token_rank = max_rank + 1
        num_rank_types = max_rank + 2
        self.embed = nn.Linear(in_features, hidden_features)
        self.rank_embed = nn.Embedding(num_rank_types, hidden_features)
        self.time_embed = nn.Embedding(max_time_steps, hidden_features)
        self.embed_drop = nn.Dropout(embed_dropout)
        self.global_token_embed = nn.Parameter(torch.randn(hidden_features) * 0.02)
        dpr = [x.item() for x in torch.linspace(0, DROP_PATH_RATE, num_layers)]
        self.layers = nn.ModuleList(
            [
                HigherOrderAttentionLayer(hidden_features, heads, num_rank_types, dropout, drop_path=dpr[i])
                for i in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_features)
        self.proj_out = nn.Linear(hidden_features, out_features)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, rank: torch.Tensor, t: torch.Tensor, batch_index: torch.Tensor
    ) -> torch.Tensor:
        N_orig = x.size(0)
        x, rank, t, batch, orig_mask = _add_global_tokens(x, rank, t, batch_index, self.token_rank)
        N = x.size(0)
        loops = torch.arange(N, device=x.device).unsqueeze(0).expand(2, -1)
        edge_index = torch.cat([edge_index, _global_token_edges(batch, ~orig_mask), loops], dim=1)
        h = self.embed(x).float()
        h[orig_mask] = h[orig_mask] + self.rank_embed(rank[orig_mask])
        h[~orig_mask] = h[~orig_mask] + self.rank_embed(rank[~orig_mask]) + self.global_token_embed
        h = h + self.time_embed(t)
        h = self.embed_drop(h)
        for layer in self.layers:
            h = layer(h, edge_index, rank, batch)
        h = self.proj_out(self.final_norm(h))
        return h[orig_mask][:N_orig]
