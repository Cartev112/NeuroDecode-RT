import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from neurodecode.train import patch_targets
from neurodecode.bench import bench_per_step
from neurodecode.quantize import quantize_int8


@torch.no_grad()
def evaluate(model, streams, cfg, device="cpu"):
    """Forecast MSE (next-patch) and class accuracy over a list of (x, labels) streams."""
    model = model.to(device).eval()
    se = 0.0; n = 0; correct = 0; total = 0
    for x, labels in streams:
        xb = x.unsqueeze(0).to(device); lb = labels.unsqueeze(0).to(device)
        fore, logits = model(xb)
        tgt = patch_targets(xb, cfg)
        se += F.mse_loss(fore[:, :-1], tgt[:, 1:], reduction="sum").item()
        n += fore[:, :-1].numel()
        correct += (logits.argmax(-1) == lb).sum().item()
        total += lb.numel()
    return {"forecast_mse": se / max(n, 1), "class_acc": correct / max(total, 1)}


def pareto(model, cfg, streams, device="cpu", **bench_kw):
    """Accuracy + per-step latency for fp32 (kv_cache) and int8 (quantized).

    Note: kv_cache/sliding-window do not change accuracy vs fp32 forward
    (equivalent / near-equivalent), so accuracy is measured on the fp32 and the
    quantized models; latency is measured per config.
    """
    rows = []
    acc_fp32 = evaluate(model, streams, cfg, device=device)
    lat_fp32 = bench_per_step(model, cfg, device=device, mode="kv_cache", **bench_kw)
    rows.append({"name": "fp32 (kv_cache)", **acc_fp32,
                 "ns_per_step": lat_fp32["ns_per_step"]})

    qmodel = quantize_int8(model)
    acc_int8 = evaluate(qmodel, streams, cfg, device="cpu")
    lat_int8 = bench_per_step(model, cfg, device="cpu", mode="quantized", **bench_kw)
    rows.append({"name": "int8 (quantized)", **acc_int8,
                 "ns_per_step": lat_int8["ns_per_step"]})
    return rows


def plot_pareto(rows, path):
    fig, ax = plt.subplots()
    for r in rows:
        ax.scatter(r["ns_per_step"] / 1e3, r["class_acc"], s=60)
        ax.annotate(r["name"], (r["ns_per_step"] / 1e3, r["class_acc"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("latency (us/step)")
    ax.set_ylabel("class accuracy")
    ax.set_title("Accuracy vs latency")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
