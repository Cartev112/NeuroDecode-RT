import torch
import torch.nn.functional as F
from neurodecode.config import Config
from neurodecode.model import NeuroDecoder
from neurodecode.quantize import quantize_int8

def test_quantized_runs_and_shapes_match():
    torch.manual_seed(0)
    cfg = Config()
    model = NeuroDecoder(cfg).eval()
    qmodel = quantize_int8(model)
    x = torch.randn(1, cfg.n_channels, cfg.patch * 16)
    with torch.no_grad():
        f0, l0 = model(x)
        f1, l1 = qmodel(x)
    assert f1.shape == f0.shape and l1.shape == l0.shape
    assert torch.isfinite(f1).all() and torch.isfinite(l1).all()
    # dynamic int8 quant of Linear layers should keep outputs well-correlated
    cos = F.cosine_similarity(f0.flatten(), f1.flatten(), dim=0)
    assert cos > 0.9
