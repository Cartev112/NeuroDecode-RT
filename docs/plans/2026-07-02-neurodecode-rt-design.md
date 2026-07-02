# NeuroDecode-RT — Design

**Date:** 2026-07-02
**Author:** Carter Whitworth (with Claude)
**Purpose:** Project 2 of the Etched Inference-Architecture application portfolio (see [InferScope](https://github.com/Cartev112/Inferscope) for Project 1). A small causal transformer that streams EEG and is optimized as a real-time *inference-serving* problem, judged against a hard latency budget. Turns a BCI/neuro background into a distinctive demonstration of the same inference-optimization skill InferScope shows on hardware.

## Shared spine with InferScope

Both projects apply the **roofline / latency-budget lens**. InferScope models transformer inference on accelerators; NeuroDecode-RT applies the same lens to a real deployment: an autoregressive transformer decoding a streaming neural signal under an edge latency budget. NeuroDecode-RT depends on the published InferScope package to reuse its roofline module — literally connecting the two repos.

## Task

Primarily **autoregressive next-patch forecasting** (the true "decode" workload: each new time-patch is one decode step, so KV-cache maps 1:1 to LLM decode), plus a **lightweight motor-imagery classification head** for a recognizable BCI accuracy metric. Small model on purpose — inference optimization is the deliverable, not SOTA accuracy.

## 1. Data pipeline

- **PhysioNet EEG Motor Movement/Imagery (EEGMMIDB)** via `mne.datasets.eegbci`. Subset (~20 subjects), motor-imagery runs, 64 channels @ 160 Hz.
- Preprocess: bandpass 0.5–40 Hz, per-channel normalization; segment into continuous sequences with a per-timestep class label (rest / left / right / fists / feet).
- **Storage: all data on D:** (C: is full). Override MNE's default `~/mne_data` (on C:) to a D: path. Cache preprocessed tensors on D: (gitignored). D: is not always mounted and unmounts on sleep — check before running; keep the download/preprocess step resumable.

## 2. Model

- Patch embedding: each token = a P=8-sample × 64-channel time-patch, linearly embedded to `d_model` (keeps sequences short).
- Decoder-only causal transformer: `d_model≈128, n_layers≈4, n_heads≈4, context≈256 patches`.
- Two heads on the causal features: **forecasting head** (predict next patch, MSE) and **class head** (cross-entropy). Joint multi-task loss.
- *Alternative considered:* per-sample tokens (no patching) — ~8× longer sequences, much heavier; rejected. Patching is the default.

## 3. Inference optimizations (the deliverable), each measured

1. **Incremental KV-cache decode vs. full recompute** — stream patch-by-patch, cache K/V; compare per-step latency to recomputing over the whole window. Centerpiece.
2. **Int8 post-training quantization** — quantize Linear layers (pragmatic dynamic quant); measure latency *and* accuracy delta (forecast MSE + class accuracy). Scoped to avoid a quantization rabbit hole.
3. **Sliding-window causal attention** — bound the KV cache to window W for unbounded streams; show bounded memory/latency vs. the accuracy cost.

## 4. Benchmarks / verdict

- **CPU (edge):** per-step latency per config vs. a real-time budget (one patch interval = 8/160 s = 50 ms; also report the tighter per-sample 6.25 ms view). Report headroom.
- **GPU:** serving throughput (steps/s and batched).
- **InferScope tie-in:** `pip install git+https://github.com/Cartev112/Inferscope`; use its roofline to project this model on a hypothetical wearable-class accelerator profile. Fallback: vendor the small roofline module if the git install is inconvenient.
- **Accuracy–latency Pareto** across optimization configs.

## 5. Testing (TDD)

- **Centerpiece correctness gate: KV-cache equivalence** — incremental decode output must match full recompute within fp tolerance (mirrors InferScope's bit-exact kernel).
- Plus: data windowing/label shapes, causal-mask correctness, forward-pass shapes, quantized-model accuracy-within-tolerance, sliding-window behavior, roofline-projection sanity.

## 6. Stack & repo

- New repo **`NeuroDecode-RT`** (`C:\Users\cwhit\Documents\GitHub\NeuroDecode-RT`, source on C:).
- **venv on D:** (CUDA PyTorch is multi-GB; C: is full). PyTorch (CUDA), MNE, NumPy, matplotlib, pytest, and InferScope as a dependency.

## Risks (managed, not hidden)

- **RTX 5050 is Blackwell `sm_120`:** stock PyTorch may lack kernels. Setup step one is verifying CUDA actually runs; if not, pin a nightly/cu128 build, else CPU-train the small model.
- **Storage:** C: full → venv + data + checkpoints on D:. D: unmounts on sleep → training resumable from D: checkpoints, runs kept short.
- **Quantization fiddliness:** scope int8 to a pragmatic, measurable dynamic-quant approach.

## Non-goals (YAGNI)

- No SOTA accuracy chase; no multi-subject leaderboard.
- No custom CUDA/Triton kernels (InferScope covers the hand-written-kernel angle).
- No multi-node/distributed serving.
