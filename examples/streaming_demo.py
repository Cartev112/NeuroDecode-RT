"""Stream an EEG window through the KV-cache decoder one patch at a time,
printing per-step latency against the real-time budget.

Run:  <venv-python> examples/streaming_demo.py
"""
import time
import torch

from neurodecode.config import Config
from neurodecode.model import NeuroDecoder
from neurodecode.synth import make_synth_stream


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuroDecoder(cfg).eval().to(device)

    x, _ = make_synth_stream(cfg, n_patches=128, seed=0)   # synthetic stream = no data dependency
    x = x.unsqueeze(0).to(device)
    patches = model.embed(x)
    cache = model.init_cache(batch=1)

    budget_ms = cfg.patch_interval_s * 1e3
    print(f"device={device}  real-time budget={budget_ms:.1f} ms/patch")
    with torch.no_grad():
        for t in range(patches.shape[1]):
            t0 = time.perf_counter()
            _, logits = model.forward_step(patches[:, t], cache, pos=t)
            if device == "cuda":
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1e3
            if t % 16 == 0:
                pred = int(logits.argmax(-1).item())
                flag = "OK" if dt_ms < budget_ms else "OVER"
                print(f"  step {t:3d}  {dt_ms:6.2f} ms  pred_class={pred}  [{flag}]")


if __name__ == "__main__":
    main()
