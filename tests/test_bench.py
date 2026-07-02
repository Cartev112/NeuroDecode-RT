from neurodecode.config import Config
from neurodecode.model import NeuroDecoder
from neurodecode.bench import bench_per_step


def test_bench_positive_and_cache_helps():
    cfg = Config(context_patches=128, n_layers=2)
    model = NeuroDecoder(cfg).eval()
    rc = bench_per_step(model, cfg, device="cpu", mode="recompute", n_steps=64, warmup=4)
    kv = bench_per_step(model, cfg, device="cpu", mode="kv_cache", n_steps=64, warmup=4)
    assert rc["ns_per_step"] > 0 and kv["ns_per_step"] > 0
    # On average per step, incremental caching beats recomputing the growing prefix
    assert kv["steps_per_s"] > rc["steps_per_s"]
