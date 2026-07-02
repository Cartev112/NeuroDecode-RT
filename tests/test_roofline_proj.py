import pytest
from neurodecode.config import Config
from neurodecode.roofline_proj import project_step_latency, HAVE_INFERSCOPE


@pytest.mark.skipif(not HAVE_INFERSCOPE, reason="inferscope not installed")
def test_projection_positive_and_scales_with_context():
    cfg = Config()
    small = project_step_latency(cfg, ctx=64)
    big = project_step_latency(cfg, ctx=1024)
    assert small > 0
    assert big > small          # longer context -> larger KV read -> higher decode latency
