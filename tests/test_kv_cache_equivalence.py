import torch
from neurodecode.config import Config
from neurodecode.model import NeuroDecoder

def test_incremental_matches_full():
    torch.manual_seed(0)
    cfg = Config(context_patches=64)
    model = NeuroDecoder(cfg).eval()
    x = torch.randn(1, cfg.n_channels, cfg.patch * 32)
    with torch.no_grad():
        full_fore, full_cls = model(x)                 # [1, 32, ...]
        patches = model.embed(x)                       # [1, 32, d_model] (no pos)
        cache = model.init_cache(batch=1)
        fore_steps, cls_steps = [], []
        for t in range(patches.shape[1]):
            f, c = model.forward_step(patches[:, t], cache, pos=t)
            fore_steps.append(f); cls_steps.append(c)
        step_fore = torch.stack(fore_steps, dim=1)
        step_cls = torch.stack(cls_steps, dim=1)
    # Not bit-identical (float op-ordering differs between batched and
    # incremental), but must be mathematically equivalent.
    max_fore = (full_fore - step_fore).abs().max().item()
    max_cls = (full_cls - step_cls).abs().max().item()
    print(f"max abs diff forecast={max_fore:.3e} class={max_cls:.3e}")
    assert torch.allclose(full_fore, step_fore, atol=1e-4)
    assert torch.allclose(full_cls, step_cls, atol=1e-4)
