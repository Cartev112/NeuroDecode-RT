import os
import glob
import torch
import torch.nn.functional as F
from neurodecode.config import Config
from neurodecode.model import NeuroDecoder


def patch_targets(x, cfg: Config):
    """x [B, C, L] -> next-patch targets [B, n_patches, C*patch] (flattened patches)."""
    B, C, L = x.shape
    n = L // cfg.patch
    xp = x[:, :, : n * cfg.patch].reshape(B, C, n, cfg.patch)
    return xp.permute(0, 2, 1, 3).reshape(B, n, C * cfg.patch)


def train_step(model, opt, x, labels, cfg: Config, lam: float = 0.5):
    """One optimization step of joint MSE(next-patch forecast) + lam*CE(class)."""
    model.train()
    fore, logits = model(x)                        # [B,T,C*patch], [B,T,K]
    tgt = patch_targets(x, cfg)                     # [B,T,C*patch]
    fmse = F.mse_loss(fore[:, :-1], tgt[:, 1:])     # predict next patch
    ce = F.cross_entropy(logits.reshape(-1, cfg.n_classes), labels.reshape(-1))
    loss = fmse + lam * ce
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.item())


def save_checkpoint(model, opt, epoch, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch:04d}.pt")
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "opt": opt.state_dict()}, path)
    return path


def latest_checkpoint(ckpt_dir):
    files = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_epoch*.pt")))
    return files[-1] if files else None


def train(cfg: Config, streams, epochs=5, lr=1e-3, device="cpu",
          ckpt_dir=None, resume=True, lam=0.5):
    """streams: list of (x [C,L], labels [n_patches]). Resumable from ckpt_dir (on D:)."""
    model = NeuroDecoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start = 0
    if resume and ckpt_dir:
        last = latest_checkpoint(ckpt_dir)
        if last:
            state = torch.load(last, map_location=device)
            model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
            start = state["epoch"] + 1
    for epoch in range(start, epochs):
        model.train()
        for x, labels in streams:
            xb = x.unsqueeze(0).to(device); lb = labels.unsqueeze(0).to(device)
            train_step(model, opt, xb, lb, cfg, lam=lam)
        if ckpt_dir:
            save_checkpoint(model, opt, epoch, ckpt_dir)
    return model
