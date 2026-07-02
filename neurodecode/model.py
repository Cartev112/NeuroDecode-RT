import math
import torch, torch.nn as nn
import torch.nn.functional as F
from neurodecode.config import Config

class PatchEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.proj = nn.Conv1d(cfg.n_channels, cfg.d_model,
                              kernel_size=cfg.patch, stride=cfg.patch)
    def forward(self, x):            # x: [B, C, L]
        return self.proj(x).transpose(1, 2)   # [B, n_patches, d_model]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)

    def _split(self, t):
        B, T, _ = t.shape
        return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,hd]

    def forward(self, x):                      # x: [B, T, d]  (full, causal)
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,T,T]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)

    def step(self, x_t, cache, window=None):   # x_t: [B,1,d]; cache: {'k','v'} or Nones
        B, _, C = x_t.shape
        q, k, v = self.qkv(x_t).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)   # [B,H,1,hd]
        if cache["k"] is None:
            K, V = k, v
        else:
            K = torch.cat([cache["k"], k], dim=2)
            V = torch.cat([cache["v"], v], dim=2)
        if window is not None and K.shape[2] > window:
            K = K[:, :, -window:, :]
            V = V[:, :, -window:, :]
        cache["k"], cache["v"] = K, V
        att = (q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,1,T]
        att = att.softmax(dim=-1)
        y = (att @ V).transpose(1, 2).contiguous().view(B, 1, C)
        return self.out(y), cache


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def step(self, x_t, cache, window=None):
        a, cache = self.attn.step(self.ln1(x_t), cache, window=window)
        x_t = x_t + a
        x_t = x_t + self.mlp(self.ln2(x_t))
        return x_t, cache


class NeuroDecoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = PatchEmbed(cfg)
        self.pos_emb = nn.Embedding(cfg.context_patches, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.forecast_head = nn.Linear(cfg.d_model, cfg.n_channels * cfg.patch)
        self.class_head = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward(self, x):                      # x: [B, C, L]
        h = self.embed(x)                      # [B, T, d]  (no positional emb)
        B, T, _ = h.shape
        pos = torch.arange(T, device=x.device)
        h = h + self.pos_emb(pos)[None]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_f(h)
        return self.forecast_head(h), self.class_head(h)

    def init_cache(self, batch):
        return [{"k": None, "v": None} for _ in range(len(self.blocks))]

    def forward_step(self, token, cache, pos, window=None):   # token: [B, d_model] (no pos emb)
        h = token.unsqueeze(1) + self.pos_emb.weight[pos][None, None, :]   # [B,1,d]
        for i, blk in enumerate(self.blocks):
            h, cache[i] = blk.step(h, cache[i], window=window)
        h = self.norm_f(h)
        return self.forecast_head(h).squeeze(1), self.class_head(h).squeeze(1)
