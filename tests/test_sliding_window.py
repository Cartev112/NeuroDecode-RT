import torch
from neurodecode.config import Config
from neurodecode.model import NeuroDecoder

def _stream(model, x, window):
    patches = model.embed(x)
    cache = model.init_cache(batch=x.shape[0])
    fs, cs = [], []
    for t in range(patches.shape[1]):
        f, c = model.forward_step(patches[:, t], cache, pos=t, window=window)
        fs.append(f); cs.append(c)
    return torch.stack(fs, 1), torch.stack(cs, 1), cache

def test_large_window_matches_full():
    torch.manual_seed(0)
    cfg = Config(context_patches=64)
    model = NeuroDecoder(cfg).eval()
    x = torch.randn(1, cfg.n_channels, cfg.patch * 32)
    with torch.no_grad():
        full_fore, full_cls = model(x)
        sf, sc, _ = _stream(model, x, window=64)   # window >= seq length
    assert torch.allclose(full_fore, sf, atol=1e-4)
    assert torch.allclose(full_cls, sc, atol=1e-4)

def test_small_window_bounds_cache():
    torch.manual_seed(0)
    cfg = Config(context_patches=64)
    model = NeuroDecoder(cfg).eval()
    x = torch.randn(1, cfg.n_channels, cfg.patch * 32)   # 32 patches
    W = 8
    with torch.no_grad():
        _, _, cache = _stream(model, x, window=W)
    # every layer's cache holds at most W entries after streaming 32 patches
    for layer_cache in cache:
        assert layer_cache["k"].shape[2] == W
        assert layer_cache["v"].shape[2] == W
