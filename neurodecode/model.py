import torch, torch.nn as nn
from neurodecode.config import Config

class PatchEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.proj = nn.Conv1d(cfg.n_channels, cfg.d_model,
                              kernel_size=cfg.patch, stride=cfg.patch)
    def forward(self, x):            # x: [B, C, L]
        return self.proj(x).transpose(1, 2)   # [B, n_patches, d_model]
