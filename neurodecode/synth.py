# neurodecode/synth.py
import numpy as np
import torch
from neurodecode.config import Config

def make_synth_stream(cfg: Config, n_patches: int, seed: int = 0):
    """Return (x, labels): x float32 [n_channels, n_patches*patch]; labels int64 [n_patches].
    Class-dependent dominant frequency so a classifier has real signal to learn."""
    rng = np.random.default_rng(seed)
    L = n_patches * cfg.patch
    t = np.arange(L) / cfg.sample_rate
    # assign each patch a class in contiguous blocks so labels are locally stable
    n_blocks = max(1, n_patches // 16)
    block_labels = rng.integers(0, cfg.n_classes, size=n_blocks)
    patch_labels = np.repeat(block_labels, int(np.ceil(n_patches / n_blocks)))[:n_patches]
    # per-class dominant frequency (Hz)
    class_freqs = np.linspace(6.0, 30.0, cfg.n_classes)
    sample_labels = np.repeat(patch_labels, cfg.patch)[:L]
    x = np.zeros((cfg.n_channels, L), dtype=np.float32)
    for ch in range(cfg.n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        freq = class_freqs[sample_labels]
        sig = np.sin(2 * np.pi * freq * t + phase)
        noise = rng.standard_normal(L) * 0.5
        x[ch] = (sig + noise).astype(np.float32)
    # per-channel normalize
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    return torch.from_numpy(x), torch.from_numpy(patch_labels.astype(np.int64))
