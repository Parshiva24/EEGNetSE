"""Inference: reproduce the paper's accuracies from saved checkpoints.

No training happens here -- each checkpoint in ``trained_model/`` is loaded and
evaluated on its subject's held-out online session.

    python -m eegnetse.infer                 # base + SE+Dense + SE-only tables
    python -m eegnetse.infer --config se_dense
    python -m eegnetse.infer --model-dir outputs   # evaluate a freshly trained set
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from eegnetse.core import (
    BANDPASS, DATA_DIR, MODEL_DIR, ONLINE_SUBJECTS,
    build_model, get_device, make_loader, online_session,
)

BASE_NAME = "basedEEGNetSE_model.pth"


def _tuned_name(subject: int, dense: bool) -> str:
    return f"tuned_sub{subject:02d}_SETrue_Dense{dense}_conv2dFalse.pth"


def evaluate_checkpoint(ckpt_path, subject, data_dir=DATA_DIR, device=None, fcut=BANDPASS) -> float:
    """Load one checkpoint and return its online-session accuracy (percent)."""
    device = device or get_device()
    model = build_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    _, _, X_online, Y_online = online_session(data_dir, subject)
    loader = make_loader(X_online, Y_online, batch_size=len(Y_online), fcut=fcut)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).squeeze()
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total


def _run(title, path_fn, subjects, data_dir, device, fcut):
    print(f"\n== {title} ==")
    accs = {}
    for s in subjects:
        p = path_fn(s)
        if not os.path.exists(p):
            print(f"  S{s:02d}: MISSING {os.path.basename(p)}")
            continue
        accs[s] = evaluate_checkpoint(p, s, data_dir, device, fcut)
        print(f"  S{s:02d}: {accs[s]:6.2f}%")
    if accs:
        v = np.array(list(accs.values()))
        lo = min(accs, key=accs.get)
        kept = np.array([a for k, a in accs.items() if k != lo])
        print(f"  mean {v.mean():.2f} +/- {v.std():.2f}%  (n={len(v)})   "
              f"drop-lowest(S{lo:02d}) {kept.mean():.2f} +/- {kept.std():.2f}%")
    return accs


def main(argv=None):
    p = argparse.ArgumentParser(description="Reproduce EEGNet-SE accuracies from saved checkpoints.")
    p.add_argument("--model-dir", default=str(MODEL_DIR))
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--config", choices=["all", "base", "se_dense", "se"], default="all",
                   help="which checkpoint family to evaluate (default: all)")
    p.add_argument("--band", default=None, help="override band-pass, e.g. '0.5,90'")
    args = p.parse_args(argv)

    fcut = tuple(float(x) for x in args.band.split(",")) if args.band else BANDPASS
    device = get_device()
    md, subs = Path(args.model_dir), list(ONLINE_SUBJECTS)
    print(f"device={device}  band={fcut}  model_dir={md}")

    if args.config in ("all", "base"):
        _run("Base as inferencer (subject-independent)",
             lambda s: str(md / BASE_NAME), subs, args.data_dir, device, fcut)
    if args.config in ("all", "se_dense"):
        _run("SE + Dense fine-tuned (proposed)",
             lambda s: str(md / _tuned_name(s, True)), subs, args.data_dir, device, fcut)
    if args.config in ("all", "se"):
        _run("SE only fine-tuned",
             lambda s: str(md / _tuned_name(s, False)), subs, args.data_dir, device, fcut)


if __name__ == "__main__":
    main()
