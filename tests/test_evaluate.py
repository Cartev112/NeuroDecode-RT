import matplotlib
matplotlib.use("Agg")
import math
from neurodecode.config import Config
from neurodecode.model import NeuroDecoder
from neurodecode.synth import make_synth_stream
from neurodecode.evaluate import evaluate, pareto, plot_pareto

def _streams(cfg, k=2):
    return [make_synth_stream(cfg, n_patches=48, seed=i) for i in range(k)]

def test_evaluate_returns_finite_metrics():
    cfg = Config(context_patches=64, n_layers=2)
    model = NeuroDecoder(cfg).eval()
    m = evaluate(model, _streams(cfg), cfg)
    assert set(m) >= {"forecast_mse", "class_acc"}
    assert math.isfinite(m["forecast_mse"])
    assert 0.0 <= m["class_acc"] <= 1.0

def test_pareto_rows_and_plot(tmp_path):
    cfg = Config(context_patches=64, n_layers=2)
    model = NeuroDecoder(cfg).eval()
    rows = pareto(model, cfg, _streams(cfg), n_steps=16, warmup=2)
    assert len(rows) >= 2
    for r in rows:
        assert math.isfinite(r["class_acc"]) and math.isfinite(r["ns_per_step"])
    out = tmp_path / "pareto.png"
    plot_pareto(rows, str(out))
    assert out.exists()
