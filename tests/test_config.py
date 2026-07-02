from neurodecode.config import Config

def test_defaults():
    c = Config()
    assert c.sample_rate == 160
    assert c.n_channels == 64
    assert c.patch == 8
    assert c.d_model == 128 and c.n_layers == 4 and c.n_heads == 4
    assert c.context_patches == 256
    assert c.n_classes >= 2

def test_patch_interval_seconds():
    c = Config()
    assert abs(c.patch_interval_s - 8/160) < 1e-9
