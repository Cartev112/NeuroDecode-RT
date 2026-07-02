import torch
from neurodecode.config import Config
from neurodecode.model import NeuroDecoder
from neurodecode.synth import make_synth_stream
from neurodecode.train import train_step

def test_train_step_reduces_loss():
    torch.manual_seed(0)
    cfg = Config(context_patches=64, n_layers=2)
    model = NeuroDecoder(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, labels = make_synth_stream(cfg, n_patches=64, seed=0)
    x = x.unsqueeze(0)            # [1, C, L]
    labels = labels.unsqueeze(0)  # [1, n_patches]
    losses = [train_step(model, opt, x, labels, cfg) for _ in range(5)]
    assert losses[-1] < losses[0]
