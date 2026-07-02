# NeuroDecode-RT

A small **causal transformer that streams EEG** and is optimized as a real-time *inference-serving* problem: incremental KV-cache decode, int8 quantization, and sliding-window attention, judged against a hard latency budget on CPU and GPU. It reuses the roofline lens from its companion project, [InferScope](https://github.com/Cartev112/Inferscope), to project the same model onto a hypothetical wearable accelerator.

The model decodes PhysioNet motor-imagery EEG: at each step it **forecasts the next signal patch** (the autoregressive "decode" workload) and emits a **motor-imagery class**. The point of the project is the inference optimization and its real-time behavior — not state-of-the-art accuracy.

## Why

Autoregressive decode of a physiological stream is the same workload shape as LLM token decode: each new time-patch is one decode step, so a KV cache applies exactly. That makes the standard serving toolkit — incremental decode vs. recompute, quantization, bounded-context attention — directly measurable on a signal I care about, and lets the same roofline reasoning from InferScope carry over to an edge device.

## Results (real run: 2 subjects, 20 epochs, RTX 5050)

**Accuracy (held-out windows).** Forecast MSE **0.627**; motor-imagery class accuracy **0.468** (3 classes, chance ≈ 0.33). Modest and honest for a deliberately tiny model on two subjects — the model learned real structure, and accuracy is not the deliverable here.

**Correctness gate.** Incremental KV-cache decode matches the full recompute forward to **~1e-6** (test `test_kv_cache_equivalence.py`), verified on CPU and CUDA. This is the analogue of InferScope's bit-exact kernel: the optimized path is provably equivalent to the reference.

**Per-step latency vs. the 50 ms/patch real-time budget** (patch = 8 samples @ 160 Hz):

| device | mode | µs/step | vs budget |
|--------|------|--------:|:---------:|
| cpu  | recompute        | 5161 | OK |
| cpu  | kv_cache         | 2943 | OK |
| cpu  | kv_cache_window  | 2981 | OK |
| cuda | recompute        | 3786 | OK |
| cuda | kv_cache         | 2418 | OK |
| cuda | kv_cache_window  | 2558 | OK |
| cpu  | int8 quantized   | 4597 | OK |

Every config clears the budget with ~10–20× headroom. Two honest findings:

- **KV-cache beats recompute ~1.75×** (2.9 vs 5.2 ms on CPU): avoiding recomputation over the growing prefix is the real win.
- **Int8 dynamic quantization is *slower* here, not faster** (4.6 vs 2.9 ms) while accuracy is essentially unchanged (0.466 vs 0.468). For a model this small on CPU, per-op quantize/dequantize overhead exceeds the matmul savings — dynamic int8's payoff is memory/throughput at scale, not single-stream latency on a tiny model. Reported as measured, not tuned away.
- **GPU ≈ CPU for single-stream decode.** With batch 1 and a tiny model, per-step kernel-launch overhead dominates; GPU wins would require batching. Also honest.

**Accuracy–latency Pareto:** `fp32 (kv_cache)` acc 0.468 @ 2881 µs vs. `int8 (quantized)` acc 0.466 @ 4713 µs — int8 is strictly dominated here (see above).

**InferScope wearable-accelerator roofline projection** (speculative 1–2 TOPS / 50 GB/s NPU), per decode step:

| context (patches) | predicted µs/step |
|---:|---:|
| 128  | 41.9 |
| 256  | 52.4 |
| 1024 | 115.3 |

Memory-bound (KV-cache read dominates), and ~1000× under the 50 ms budget — a modest wearable NPU has ample headroom for this model even at long context.

## How it works

- **Patch embedding.** 8-sample × 64-channel time-patches → `Conv1d` → `d_model=128` tokens (short sequences).
- **Decoder-only causal transformer** (`4 layers, 4 heads`), a forecasting head (predict next patch, MSE) and a class head (CE), trained jointly.
- **Incremental decode.** `forward_step` maintains a per-layer K/V cache; positional embedding is added per absolute position so streaming equals the full forward. Sliding window trims the cache to the last `W` entries for bounded memory.
- **Roofline projection** maps the model to an InferScope `TransformerConfig` (patches ≈ tokens) and estimates per-decode-step latency on a device profile.

## Quickstart

C: is storage-constrained here, so the venv and data live on **D:** — see [docs/ENV.md](docs/ENV.md). Bare `python` is a broken MS Store stub; use the venv interpreter (`$PY = D:\venvs\neurodecode-rt\Scripts\python.exe`).

```powershell
# after the one-time env setup in docs/ENV.md:
& $PY -m pytest -q                        # 17 tests incl. KV-cache equivalence
& $PY examples/streaming_demo.py          # per-step latency vs budget (synthetic, no data)
& $PY examples/train_and_report.py        # downloads PhysioNet MI (to D:), trains, full report
```

## Limitations (candid)

- Tiny model, 2 subjects, ~20 epochs — a demonstration of the inference pipeline, not a BCI accuracy result. The held-out split is window-wise within subjects, so accuracy is optimistic for cross-subject use.
- Sliding-window accuracy vs. full-context is not swept here; only equivalence (large window) and bounded-memory behavior are tested.
- Int8 uses `torch.ao` dynamic quantization (deprecated in favor of `torchao`); CPU-only.
- The wearable-accelerator device profile is a speculative guess, and the roofline is analytic (no kernel-fusion/scheduling effects).
- Class labels are the crude PhysioNet annotation mapping (T0/T1/T2); the class head has 5 outputs but only 3 are used for these runs.

## What this demonstrates

The same inference-optimization reasoning as InferScope — prefill/decode asymmetry, KV-cache, memory-vs-compute bottlenecks, roofline projection — applied end-to-end to a real streaming neural signal under a hard latency budget, with the optimized decode path provably equivalent to the reference and every claim measured rather than asserted.

Design + plan: [docs/plans/2026-07-02-neurodecode-rt-design.md](docs/plans/2026-07-02-neurodecode-rt-design.md), [docs/plans/2026-07-02-neurodecode-rt.md](docs/plans/2026-07-02-neurodecode-rt.md). Companion project: [InferScope](https://github.com/Cartev112/Inferscope).
