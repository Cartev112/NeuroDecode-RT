# NeuroDecode-RT Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement this plan task-by-task.

**Goal:** Build a small causal transformer that autoregressively forecasts streaming EEG (plus a motor-imagery class head), then optimize its inference (incremental KV-cache, int8 quantization, sliding-window attention) and prove it runs under a real-time budget on CPU (edge) and GPU, tying back to InferScope's roofline.

**Architecture:** Decoder-only transformer over 8-sample EEG time-patches. Forecasting head (MSE) + class head (CE), joint loss. Inference optimizations are the deliverable; correctness is gated by a KV-cache-equivalence test (incremental decode == full recompute). Benchmarks compare optimization configs on CPU and GPU against a per-patch latency budget; InferScope projects a wearable-accelerator roofline.

**Tech Stack:** Python 3.11+, PyTorch (CUDA build for `sm_120`, else CPU), MNE (dataset), NumPy, matplotlib, pytest, and `inferscope` (from GitHub).

---

## Conventions & environment (READ FIRST)

- **Storage: C: is full.** The git repo (source) is on C: at `C:\Users\cwhit\Documents\GitHub\NeuroDecode-RT`, but the **venv, dataset, cache, and checkpoints go on D:**. D: is not always mounted and unmounts on sleep — check `Get-PSDrive D` before any step that touches it; keep training resumable.
- **venv path:** `D:\venvs\neurodecode-rt`. Interpreter referenced below as **`$PY`** = `D:\venvs\neurodecode-rt\Scripts\python.exe`. (Bare `python` is a broken MS Store stub on this machine.)
- **Data root:** `$DATA` = `D:\neurodecode-rt-data` (MNE download dir, preprocessed cache, checkpoints all under here).
- Run tests with `& $PY -m pytest`. Commit after every green test. Conventional Commits. Author `Carter Whitworth <cwhit1129@gmail.com>`, co-author trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Tasks are grouped:** Tasks 1–9 need only torch (testable on synthetic tensors — no dataset/GPU required). Tasks 10–13 need the dataset (D:) and, for real runs, the GPU. Each task marks its requirement.

---

### Task 1: Environment + scaffold  [requires D: mounted]

**Files:** `pyproject.toml`, `neurodecode/__init__.py`, `tests/test_smoke.py`, `.gitignore`

**Steps:**
1. Verify D: mounted (`Get-PSDrive D`). Create venv on D:: `py -m venv D:\venvs\neurodecode-rt`.
2. Install deps into `$PY`: first probe CUDA support for the RTX 5050 (`sm_120`). Try stable CUDA wheel; if `torch.cuda.is_available()` is True but a tiny `x = torch.randn(8,device='cuda'); (x@x).sum()` errors with a `sm_120`/kernel-image error, install a nightly cu128 build instead. If CUDA can't be made to work, proceed CPU-only (model is small) and note it. Then install `mne numpy matplotlib pytest` and `pip install "git+https://github.com/Cartev112/Inferscope"`.
3. Write `pyproject.toml` (name `neurodecode`, deps listed but torch/inferscope may be installed manually per above), `neurodecode/__init__.py` with `__version__="0.1.0"`, `.gitignore` (`__pycache__/`, `*.pyc`, `.pytest_cache/`, `*.egg-info/`, `build/`), and `tests/test_smoke.py` asserting `neurodecode.__version__` is a str.
4. `& $PY -m pip install -e .` ; `& $PY -m pytest -v` → PASS.
5. Record in `README.md` (stub) or a `docs/ENV.md` the exact torch build that worked and whether CUDA is active.
6. Commit: `chore: scaffold neurodecode package + env`.

**Note for executor:** if D: is not mounted, STOP and ask the user to connect it — do not create the venv on C:.

---

### Task 2: Config module  [torch not required]

**Files:** Create `neurodecode/config.py`; Test `tests/test_config.py`

**Step 1: failing test**
```python
# tests/test_config.py
from neurodecode.config import Config

def test_defaults():
    c = Config()
    assert c.sample_rate == 160
    assert c.n_channels == 64
    assert c.patch == 8
    assert c.d_model == 128 and c.n_layers == 4 and c.n_heads == 4
    assert c.context_patches == 256
    assert c.n_classes >= 2

def test_patch_interval_seconds():
    c = Config()
    assert abs(c.patch_interval_s - 8/160) < 1e-9   # 50 ms budget
```

**Step 3: implementation**
```python
# neurodecode/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    sample_rate: int = 160
    n_channels: int = 64
    patch: int = 8
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    context_patches: int = 256
    n_classes: int = 5          # rest / left / right / fists / feet
    data_root: str = r"D:\neurodecode-rt-data"

    @property
    def patch_interval_s(self) -> float:
        return self.patch / self.sample_rate
```
Commit: `feat: config with model + timing defaults`.

---

### Task 3: Patch embedding  [torch]

**Files:** Create `neurodecode/model.py`; Test `tests/test_patch_embed.py`

Embed a stream `[B, n_channels, L]` (L a multiple of `patch`) into patch tokens `[B, n_patches, d_model]` via a `Conv1d(n_channels, d_model, kernel_size=patch, stride=patch)`.

**Test:** random `x = torch.randn(2, 64, 8*10)`; `PatchEmbed(cfg)(x).shape == (2, 10, 128)`.

**Implementation:**
```python
# neurodecode/model.py
import torch, torch.nn as nn
from neurodecode.config import Config

class PatchEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.proj = nn.Conv1d(cfg.n_channels, cfg.d_model,
                              kernel_size=cfg.patch, stride=cfg.patch)
    def forward(self, x):            # x: [B, C, L]
        return self.proj(x).transpose(1, 2)   # [B, n_patches, d_model]
```
Commit: `feat: EEG patch embedding`.

---

### Task 4: Causal transformer + heads (no cache)  [torch]

**Files:** Modify `neurodecode/model.py`; Test `tests/test_model_forward.py`

Add a decoder-only stack (pre-norm, causal multi-head self-attention, MLP), a forecasting head (`Linear(d_model, n_channels*patch)` predicting the next patch), and a class head (`Linear(d_model, n_classes)`). `forward(x_patches)` returns `(next_patch_pred, class_logits)` each aligned per time-step.

**Tests (both important):**
1. **Shapes:** input `[2, 64, 8*16]` → embed → model returns forecast `[2, 16, 64*8]` and logits `[2, 16, n_classes]`.
2. **Causality:** perturbing input patch at position `t` must NOT change outputs at positions `< t`. Build sequence, run, clone, modify one future patch, re-run, assert earlier outputs identical (`torch.allclose`).

Provide complete module code (attention with a causal mask via `torch.triu`, RoPE or learned positional embedding — use a simple learned positional embedding indexed by patch position, with a max of `context_patches`). Keep it standard and readable.
Commit: `feat: causal transformer with forecast + class heads`.

---

### Task 5: Incremental KV-cache decode + equivalence  [torch] — CENTERPIECE

**Files:** Modify `neurodecode/model.py` (add `forward_step`/cache); Test `tests/test_kv_cache_equivalence.py`

Implement per-layer KV cache: `init_cache(batch)` and `forward_step(patch_token, cache, pos)` that appends the new K/V and attends over the cached history, returning the same per-step outputs as the full forward.

**Step 1: failing test (the correctness gate)**
```python
# tests/test_kv_cache_equivalence.py
import torch
from neurodecode.config import Config
from neurodecode.model import NeuroDecoder

def test_incremental_matches_full():
    torch.manual_seed(0)
    cfg = Config(context_patches=64)
    model = NeuroDecoder(cfg).eval()
    x = torch.randn(1, cfg.n_channels, cfg.patch * 32)
    with torch.no_grad():
        full_fore, full_cls = model(x)                 # [1, 32, ...]
        patches = model.embed(x)                       # [1, 32, d_model]
        cache = model.init_cache(batch=1)
        step_fore, step_cls = [], []
        for t in range(patches.shape[1]):
            f, c = model.forward_step(patches[:, t], cache, pos=t)
            step_fore.append(f); step_cls.append(c)
        step_fore = torch.stack(step_fore, dim=1)
        step_cls = torch.stack(step_cls, dim=1)
    assert torch.allclose(full_fore, step_fore, atol=1e-5)
    assert torch.allclose(full_cls, step_cls, atol=1e-5)
```
Run → FAIL (no `forward_step`). Implement cache; run → PASS. This mirrors InferScope's bit-exact kernel gate.
Commit: `feat: incremental KV-cache decode (equivalent to full recompute)`.

---

### Task 6: Sliding-window causal attention  [torch]

**Files:** Modify `neurodecode/model.py`; Test `tests/test_sliding_window.py`

Add `window` param to the cache/attention: when set, `forward_step` keeps only the last `W` K/V entries (bounded memory for unbounded streams).

**Tests:**
1. With `W >= context`, streaming output equals the unbounded-cache output (allclose).
2. With small `W`, output at step `t` is unchanged by inputs older than `t-W` (perturb an out-of-window past patch → later output identical).
Commit: `feat: sliding-window bounded KV cache`.

---

### Task 7: Int8 post-training quantization  [torch]

**Files:** Create `neurodecode/quantize.py`; Test `tests/test_quantize.py`

Use `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` (dynamic quant of Linear — pragmatic, CPU-only inference). Expose `quantize_int8(model) -> model`.

**Tests:**
1. Quantized model forward returns same shapes.
2. On a fixed random input, quantized output is close to fp32 within a loose tolerance (e.g. relative MSE < 5%). (Correctness-of-shape + sanity, not exactness.)
Commit: `feat: int8 dynamic quantization wrapper`.

---

### Task 8: Latency benchmark harness (CPU + GPU)  [torch; GPU optional]

**Files:** Create `neurodecode/bench.py`; Test `tests/test_bench.py`

`bench_per_step(model, cfg, device, mode)` where `mode in {"recompute","kv_cache","kv_cache_window","quantized"}`. Warm up, time N steps, return `ns_per_step` and `steps_per_s`. `bench_all(...)` returns a dict over modes; GPU entries skipped if `torch.cuda.is_available()` is False.

**Test:** on a tiny model + random stream, `bench_per_step` returns positive `ns_per_step`; `kv_cache` mode is not slower than `recompute` at a long context (sanity: caching helps). Keep N small for test speed.
Commit: `feat: per-step latency benchmark harness`.

---

### Task 9: Roofline projection via InferScope  [torch + inferscope]

**Files:** Create `neurodecode/roofline_proj.py`; Test `tests/test_roofline_proj.py`

Import `inferscope` (`from inferscope.device import Device`, `from inferscope.roofline import roofline_latency`). Build a `TransformerConfig`-like description of this model and a hypothetical **wearable-class accelerator** `Device`, and predict per-decode-step latency; expose `project_step_latency(cfg, device) -> seconds`.

**Test:** `project_step_latency(...) > 0`; and it scales with `context` (longer context → more KV read → higher predicted latency). If `inferscope` import fails, `pytest.skip` with a clear message (fallback: a vendored `_roofline.py`).
Commit: `feat: wearable-accelerator roofline projection via InferScope`.

---

### Task 10: Data pipeline  [requires D: + MNE download]

**Files:** Create `neurodecode/data.py`, `neurodecode/synth.py`; Test `tests/test_data.py`

- `neurodecode/synth.py`: a **synthetic** multichannel EEG generator with class-dependent spectral content — used for all unit tests and for CI-style runs without the download. `make_synth_stream(cfg, n_patches, seed) -> (x[C,L], labels[n_patches])`.
- `neurodecode/data.py`: `load_eegbci(cfg, subjects)` downloading via `mne.datasets.eegbci.load_data(..., path=cfg.data_root)` (→ D:), bandpass 0.5–40 Hz, per-channel normalize, segment into `[C, L]` streams with per-patch labels; cache preprocessed `.pt` tensors under `cfg.data_root` (skip re-preprocess if cache exists — resumable).

**Tests (use synth, no download):** windowing/label shapes correct; `make_synth_stream` returns `x` shape `[64, n_patches*8]` and `labels` shape `[n_patches]`. The real `load_eegbci` is covered by an integration test marked `@pytest.mark.skipif(not Path(cfg.data_root).exists(), ...)`.
Commit: `feat: EEG data pipeline (MNE) + synthetic generator`.

---

### Task 11: Training loop  [requires D: + GPU for real runs]

**Files:** Create `neurodecode/train.py`; Test `tests/test_train_step.py`

Joint loss `L = mse(forecast, next_patch) + λ·ce(class_logits, labels)`. `train(cfg, data, epochs, ckpt_dir)` with checkpointing to `cfg.data_root` (resumable: load latest checkpoint if present) and short epochs.

**Test (synthetic, CPU, no dataset):** one `train_step` on a tiny synthetic batch reduces the loss (run 5 steps, assert loss[-1] < loss[0]). This verifies the training wiring without the dataset/GPU.
Commit: `feat: joint forecast+class training loop (resumable checkpoints)`.

---

### Task 12: Accuracy + Pareto evaluation  [torch; real numbers need trained ckpt]

**Files:** Create `neurodecode/evaluate.py`; Test `tests/test_evaluate.py`

`evaluate(model, data, cfg) -> {"forecast_mse":..., "class_acc":...}` and `pareto(configs) -> rows` combining accuracy with `bench` latency. Plot accuracy vs latency across `{fp32, quantized, kv_cache, sliding W=...}`.

**Test (synthetic/tiny model):** `evaluate` returns a metrics dict with finite values; `pareto` returns one row per config. Plot smoke-tested with Agg backend.
Commit: `feat: accuracy + accuracy-latency Pareto evaluation`.

---

### Task 13: Example scripts + README  [real numbers need D:+GPU run]

**Files:** Create `examples/train_and_report.py`, `examples/streaming_demo.py`, `README.md`

- `train_and_report.py`: download/prep (D:), train (GPU), evaluate, print latency table (CPU+GPU across modes), save Pareto + roofline plots.
- `streaming_demo.py`: stream a signal through the KV-cache model, printing per-step latency vs the 50 ms budget.
- `README.md`: what/why, quickstart (with `$PY` / D: notes), results (KV-cache equivalence, CPU/GPU latency table, Pareto, InferScope roofline projection), how it works, candid limitations (small model; synthetic option; dynamic-quant scope; sm_120/CPU caveat; roofline is analytic), and the closing narrative tying to InferScope. Link both design docs and the InferScope repo.

Fill real numbers by running the examples once D:+GPU are available.
Commit: `docs: README with results and InferScope tie-in`.

---

## Done criteria

- Tasks 1–9: `& $PY -m pytest` green (KV-cache equivalence and causality tests passing) — provable without the dataset.
- Tasks 10–13: data pipeline + training + evaluation run end-to-end once D: is mounted and a CUDA (or CPU) torch is working; README reports real CPU/GPU latency, the accuracy–latency Pareto, and the InferScope roofline projection.
