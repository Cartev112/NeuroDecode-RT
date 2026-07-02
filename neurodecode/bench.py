import time
import torch
from neurodecode.quantize import quantize_int8


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def bench_per_step(model, cfg, device="cpu", mode="kv_cache",
                   n_steps=64, warmup=8, window=32):
    """Average per-step latency for one streaming inference mode.

    modes:
      recompute        - naive: re-run full forward over the growing prefix each step
      kv_cache         - incremental decode with unbounded KV cache
      kv_cache_window  - incremental decode with sliding window
      quantized        - int8 dynamic-quant model, incremental decode (CPU only)
    """
    total = n_steps + warmup
    if mode == "quantized":
        run_model = quantize_int8(model)
        device = "cpu"
    else:
        run_model = model.to(device).eval()
    x = torch.randn(1, cfg.n_channels, cfg.patch * total, device=device)

    def one_pass():
        if mode == "recompute":
            for t in range(total):
                run_model(x[:, :, : (t + 1) * cfg.patch])
        else:
            patches = run_model.embed(x)
            cache = run_model.init_cache(batch=1)
            w = window if mode == "kv_cache_window" else None
            for t in range(total):
                run_model.forward_step(patches[:, t], cache, pos=t, window=w)

    one_pass(); _sync(device)                 # warmup pass
    t0 = time.perf_counter()
    one_pass(); _sync(device)                 # timed pass
    t1 = time.perf_counter()
    ns_per_step = (t1 - t0) * 1e9 / total
    return {"mode": mode, "device": str(device),
            "ns_per_step": ns_per_step, "steps_per_s": 1e9 / ns_per_step}


def bench_all(model, cfg, devices=None, **kw):
    """Benchmark all modes on the given devices (cpu + cuda if available)."""
    if devices is None:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    results = {}
    for device in devices:
        for mode in ("recompute", "kv_cache", "kv_cache_window"):
            results[(device, mode)] = bench_per_step(model, cfg, device=device, mode=mode, **kw)
    results[("cpu", "quantized")] = bench_per_step(model, cfg, device="cpu", mode="quantized", **kw)
    return results
