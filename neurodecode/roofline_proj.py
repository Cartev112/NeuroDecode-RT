from neurodecode.config import Config

try:
    from inferscope.device import Device
    from inferscope.model_config import TransformerConfig
    from inferscope.engine import estimate
    HAVE_INFERSCOPE = True
except Exception:
    HAVE_INFERSCOPE = False


# Speculative wearable-class edge NPU (explicit guess, not a real device).
def wearable_device():
    return Device(
        name="wearable-npu (speculative)",
        peak_flops={"fp16": 1e12, "int8": 2e12},   # ~1-2 TOPS
        mem_bandwidth=5e10,                          # 50 GB/s
        sram_bytes=2e6,                              # 2 MB
    )


def _to_inferscope_cfg(cfg: Config) -> "TransformerConfig":
    # patches are the "tokens"; standard (non-gated) MLP with d_ff = 4*d_model;
    # MHA so n_kv_heads == n_heads; "vocab" is the forecast output width.
    return TransformerConfig(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_heads, d_ff=4 * cfg.d_model,
        vocab_size=cfg.n_channels * cfg.patch, gated_mlp=False,
    )


def project_step_latency(cfg: Config, ctx: int, device=None, dtype="fp16") -> float:
    """Roofline-projected latency (seconds) for ONE decode step (one patch) on
    a wearable accelerator, via the InferScope model."""
    if not HAVE_INFERSCOPE:
        raise RuntimeError("inferscope is not installed")
    device = device or wearable_device()
    est = estimate(_to_inferscope_cfg(cfg), device,
                   batch=1, seq=1, ctx=ctx, regime="decode", dtype=dtype)
    return est.latency_s
