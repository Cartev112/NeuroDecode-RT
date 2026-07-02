import torch
from neurodecode.config import Config
from neurodecode.model import NeuroDecoder

def test_forward_shapes():
    cfg = Config()
    model = NeuroDecoder(cfg).eval()
    x = torch.randn(2, cfg.n_channels, cfg.patch * 16)   # [B, C, L]
    forecast, logits = model(x)
    assert forecast.shape == (2, 16, cfg.n_channels * cfg.patch)
    assert logits.shape == (2, 16, cfg.n_classes)

def test_causality_future_does_not_affect_past():
    torch.manual_seed(0)
    cfg = Config()
    model = NeuroDecoder(cfg).eval()
    x = torch.randn(1, cfg.n_channels, cfg.patch * 16)
    with torch.no_grad():
        f1, l1 = model(x)
        x2 = x.clone()
        # perturb the LAST patch's raw samples (a "future" position)
        x2[:, :, cfg.patch * 15:] += 5.0
        f2, l2 = model(x2)
    # outputs at positions < 15 must be unchanged (patches are non-overlapping,
    # attention is causal)
    assert torch.allclose(f1[:, :15], f2[:, :15], atol=1e-5)
    assert torch.allclose(l1[:, :15], l2[:, :15], atol=1e-5)
