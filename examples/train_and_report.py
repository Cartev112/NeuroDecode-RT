"""Train NeuroDecode-RT on PhysioNet motor imagery and report the full result:
test forecast MSE + class accuracy, CPU/GPU per-step latency across optimization
modes vs the real-time budget, an accuracy-latency Pareto, and an InferScope
wearable-accelerator roofline projection.

Run:  <venv-python> examples/train_and_report.py
Requires D: mounted (data + checkpoints live there).
"""
import os
import json
import time
import torch

from neurodecode.config import Config
from neurodecode.data import load_eegbci
from neurodecode.model import NeuroDecoder
from neurodecode.train import train_step, save_checkpoint
from neurodecode.evaluate import evaluate, pareto, plot_pareto
from neurodecode.bench import bench_all
from neurodecode.roofline_proj import project_step_latency, HAVE_INFERSCOPE

SUBJECTS = (1, 2)
WIN = 128          # patches per training window (< context_patches = 256)
EPOCHS = 20
LR = 1e-3


def chunk(x, labels, cfg, win):
    """Slice a long (x [C,L], labels [n_patches]) stream into fixed patch windows."""
    P = cfg.patch
    n_patches = x.shape[1] // P
    out = []
    for sp in range(0, n_patches - win + 1, win):
        s = sp * P
        out.append((x[:, s: s + win * P].contiguous(), labels[sp: sp + win].contiguous()))
    return out


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    windows = []
    for subj in SUBJECTS:
        x, labels = load_eegbci(cfg, subjects=(subj,))
        windows += chunk(x, labels, cfg, WIN)
    print(f"subjects={SUBJECTS} windows={len(windows)} (win={WIN} patches)")

    n_test = max(1, len(windows) // 5)
    train_w, test_w = windows[:-n_test], windows[-n_test:]

    model = NeuroDecoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ckpt_dir = os.path.join(cfg.data_root, "checkpoints")
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for x, labels in train_w:
            xb = x.unsqueeze(0).to(device)
            lb = labels.unsqueeze(0).to(device)
            tot += train_step(model, opt, xb, lb, cfg)
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:2d}  avg_loss {tot/len(train_w):.4f}  {time.time()-t0:.1f}s")
    save_checkpoint(model, opt, EPOCHS - 1, ckpt_dir)

    m = evaluate(model, test_w, cfg, device=device)
    print(f"TEST  forecast_mse={m['forecast_mse']:.4f}  class_acc={m['class_acc']:.3f}")

    budget_ms = cfg.patch_interval_s * 1e3
    print(f"\nreal-time budget = {budget_ms:.1f} ms/patch  (patch = {cfg.patch}/{cfg.sample_rate}s)")
    print("=== per-step latency ===")
    res = bench_all(model, cfg, n_steps=64, warmup=8)
    for (dev, mode), r in res.items():
        ok = "OK" if r["ns_per_step"] / 1e6 < budget_ms else "OVER"
        print(f"  {dev:5s} {mode:16s} {r['ns_per_step']/1e3:9.1f} us/step  {r['steps_per_s']:8.0f}/s  [{ok}]")

    rows = pareto(model, cfg, test_w, n_steps=64, warmup=8)
    plot_pareto(rows, "pareto.png")
    print("=== accuracy-latency Pareto ===")
    for r in rows:
        print(f"  {r['name']:18s} acc={r['class_acc']:.3f}  mse={r['forecast_mse']:.4f}  {r['ns_per_step']/1e3:.1f} us/step")

    if HAVE_INFERSCOPE:
        print("=== InferScope wearable-accelerator roofline projection ===")
        for ctx in (128, 256, 1024):
            print(f"  ctx={ctx:5d}: {project_step_latency(cfg, ctx=ctx)*1e6:8.1f} us/step (predicted)")

    metrics = {
        "device": device, "subjects": list(SUBJECTS), "windows": len(windows),
        "test": m, "budget_ms": budget_ms,
        "latency_ns_per_step": {f"{d}_{mo}": res[(d, mo)]["ns_per_step"] for (d, mo) in res},
        "pareto": rows,
    }
    out = os.path.join(cfg.data_root, "report_metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
