# neurodecode/data.py
import os
import numpy as np
import torch
from neurodecode.config import Config

def labels_per_patch(sample_labels, patch: int):
    """Downsample a per-sample label array to one label per patch (majority vote).
    sample_labels: 1D int array length L (multiple of patch). Returns int64 [L//patch]."""
    import numpy as np
    sl = np.asarray(sample_labels)
    n = len(sl) // patch
    sl = sl[: n * patch].reshape(n, patch)
    # majority per row
    out = np.array([np.bincount(row).argmax() for row in sl], dtype=np.int64)
    return out

def load_eegbci(cfg: Config, subjects=(1,), runs=(4, 8, 12), use_cache=True):
    """Download (to cfg.data_root) + preprocess PhysioNet Motor Imagery via MNE.
    Returns (x, patch_labels): x float32 [n_channels, L]; patch_labels int64 [L//patch].
    Caches the preprocessed tensor to cfg.data_root; re-preprocessing is skipped if cache exists."""
    os.makedirs(cfg.data_root, exist_ok=True)
    cache = os.path.join(cfg.data_root, f"eegbci_s{'-'.join(map(str,subjects))}.pt")
    if use_cache and os.path.exists(cache):
        d = torch.load(cache)
        return d["x"], d["labels"]
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf
    raws = []
    for s in subjects:
        for f in eegbci.load_data(s, runs, path=cfg.data_root):
            raws.append(read_raw_edf(f, preload=True))
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.filter(0.5, 40.0, fir_design="firwin", verbose=False)
    data = raw.get_data().astype("float32")                 # [C, T]
    data = (data - data.mean(1, keepdims=True)) / (data.std(1, keepdims=True) + 1e-8)
    # crude per-sample labels from annotations (T0=rest=0, T1=1, T2=2); default 0
    sample_labels = np.zeros(data.shape[1], dtype=np.int64)
    sfreq = raw.info["sfreq"]
    for ann in raw.annotations:
        code = {"T0": 0, "T1": 1, "T2": 2}.get(ann["description"], 0)
        a = int(ann["onset"] * sfreq); b = a + int(ann["duration"] * sfreq)
        sample_labels[a:b] = code
    x = torch.from_numpy(data)
    labels = torch.from_numpy(labels_per_patch(sample_labels, cfg.patch))
    # trim x to a whole number of patches matching labels
    x = x[:, : labels.shape[0] * cfg.patch]
    if use_cache:
        torch.save({"x": x, "labels": labels}, cache)
    return x, labels
