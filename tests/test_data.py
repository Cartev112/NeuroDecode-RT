import numpy as np
from neurodecode.config import Config
from neurodecode.synth import make_synth_stream
from neurodecode.data import labels_per_patch

def test_synth_shapes():
    cfg = Config()
    x, labels = make_synth_stream(cfg, n_patches=40, seed=1)
    assert x.shape == (cfg.n_channels, 40 * cfg.patch)
    assert labels.shape == (40,)
    assert int(labels.min()) >= 0 and int(labels.max()) < cfg.n_classes
    assert x.dtype.is_floating_point

def test_labels_per_patch_majority():
    # patch=4; two patches: first all class 2, second majority class 1
    sl = [2,2,2,2, 1,1,1,0]
    out = labels_per_patch(sl, 4)
    assert list(out) == [2, 1]
