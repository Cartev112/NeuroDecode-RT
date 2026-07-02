# Environment

C: is storage-constrained, so the virtual environment and all data/checkpoints live on **D:**.

## Interpreter

- venv: `D:\venvs\neurodecode-rt`
- Python: `D:\venvs\neurodecode-rt\Scripts\python.exe` (referenced as `$PY`)
- Bare `python` on this machine is a non-functional Microsoft Store stub — always use `$PY` or the `py` launcher.

## Verified working build (2026-07-02)

- **torch 2.11.0+cu128**, CUDA available, device **NVIDIA GeForce RTX 5050 Laptop GPU**, compute capability **(12, 0)** = `sm_120` (Blackwell). A real GPU matmul runs — no nightly needed.
- numpy 2.5.0, mne 1.12.1, matplotlib 3.11.0, pytest 9.1.1.
- `inferscope` 0.1.0 installed from GitHub (`pip install "git+https://github.com/Cartev112/Inferscope"`).

## Reproduce

```powershell
py -m venv D:\venvs\neurodecode-rt
$PY = "D:\venvs\neurodecode-rt\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& $PY -m pip install numpy mne matplotlib pytest
& $PY -m pip install "git+https://github.com/Cartev112/Inferscope"
& $PY -m pip install -e .
```

## Data / checkpoints

- Data root: `D:\neurodecode-rt-data` (MNE download dir, preprocessed cache, checkpoints).
- D: unmounts on sleep and is not always connected — check `Get-PSDrive D` before running data/training steps, and keep training resumable from checkpoints.
