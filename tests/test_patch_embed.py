import torch
from neurodecode.config import Config
from neurodecode.model import PatchEmbed

def test_patch_embed_shape():
    cfg = Config()
    x = torch.randn(2, cfg.n_channels, cfg.patch * 10)   # [B, C, L]
    out = PatchEmbed(cfg)(x)
    assert out.shape == (2, 10, cfg.d_model)
