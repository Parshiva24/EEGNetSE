"""Full training: base pre-training and per-subject fine-tuning.

Two-stage transfer learning (as in the manuscript):

    Stage 1  pretrain_base      -> subject-independent base on S01-S07 calibration
    Stage 2  finetune_subject   -> per online subject (S08-S20), SE (+dense) layers

Run everything from the command line::

    python -m eegnetse.train --stage all                 # base + all subjects
    python -m eegnetse.train --stage base
    python -m eegnetse.train --stage finetune --no-dense  # SE-only variant

Freshly trained weights are written to ``--out-dir`` (default ``outputs/``) using
the same filenames as the committed reference set, so they never clobber
``trained_model/`` unless you point ``--out-dir`` there deliberately.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from eegnetse.core import (
    BANDPASS, DATA_DIR, ONLINE_SUBJECTS,
    build_model, calibration_dataset, get_device, make_loader,
    online_session, set_seed, set_trainable,
)

BASE_NAME = "basedEEGNetSE_model.pth"


def _tuned_name(subject: int, dense: bool) -> str:
    return f"tuned_sub{subject:02d}_SETrue_Dense{dense}_conv2dFalse.pth"


def _accuracy(outputs, targets) -> float:
    return 100.0 * (outputs.argmax(dim=1) == targets).float().mean().item()


def _evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).squeeze()
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total


def train_model(model, train_loader, val_loader, device, *, lr, max_epochs=1000,
                patience=30, min_delta=1e-3, min_epochs=0, verbose=False):
    """Adam + early stopping on validation accuracy; restore the best weights."""
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)

    best_acc, best_state, wait = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).squeeze()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_acc = _evaluate(model, val_loader, device)
        if verbose:
            print(f"  epoch {epoch + 1:4d} | val_acc {val_acc:5.2f}%")

        if val_acc > best_acc + min_delta:
            best_acc, best_state, wait = val_acc, copy.deepcopy(model.state_dict()), 0
        elif epoch + 1 >= min_epochs:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #
def pretrain_base(data_dir=DATA_DIR, device=None, *, fcut=BANDPASS, lr=1e-4,
                  max_epochs=1000, val_ratio=0.2, seed=0, verbose=False):
    device = device or get_device()
    set_seed(seed)
    X, Y = calibration_dataset(data_dir)
    Xtr, Xval, Ytr, Yval = train_test_split(
        X, Y, train_size=1 - val_ratio, random_state=seed, shuffle=True, stratify=Y)
    train_loader = make_loader(Xtr, Ytr, batch_size=32, shuffle=True, fcut=fcut)
    val_loader = make_loader(Xval, Yval, batch_size=len(Yval), fcut=fcut)

    model = build_model().to(device)
    model, acc = train_model(model, train_loader, val_loader, device,
                             lr=lr, max_epochs=max_epochs, min_epochs=100, verbose=verbose)
    return model, acc


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #
def finetune_subject(base_state, subject, data_dir=DATA_DIR, device=None, *,
                     dense=True, fcut=BANDPASS, lr=1e-3, max_epochs=1000,
                     val_ratio=0.2, seed=0, verbose=False):
    device = device or get_device()
    set_seed(seed)
    Xtr, Ytr, _, _ = online_session(data_dir, subject)   # subject calibration trials
    Xt, Xv, Yt, Yv = train_test_split(
        Xtr, Ytr, train_size=1 - val_ratio, random_state=seed, shuffle=True, stratify=Ytr)
    train_loader = make_loader(Xt, Yt, batch_size=32, shuffle=True, fcut=fcut)
    val_loader = make_loader(Xv, Yv, batch_size=len(Yv), fcut=fcut)

    model = build_model().to(device)
    model.load_state_dict(copy.deepcopy(base_state))
    groups = ["se", "dense"] if dense else ["se"]
    n_trainable = set_trainable(model, groups)
    model, acc = train_model(model, train_loader, val_loader, device,
                             lr=lr, max_epochs=max_epochs, verbose=verbose)
    return model, acc, n_trainable


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description="Train EEGNet-SE (base + per-subject fine-tuning).")
    p.add_argument("--stage", choices=["base", "finetune", "all"], default="all")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "outputs"))
    p.add_argument("--subjects", default="8-20", help="e.g. '8-20' or '8,9,10'")
    p.add_argument("--dense", dest="dense", action="store_true", default=True,
                   help="fine-tune SE + Dense (default; the proposed configuration)")
    p.add_argument("--no-dense", dest="dense", action="store_false", help="fine-tune SE only")
    p.add_argument("--band", default=None, help="override band-pass, e.g. '0.5,90'")
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    fcut = tuple(float(x) for x in args.band.split(",")) if args.band else BANDPASS
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = _parse_subjects(args.subjects)
    print(f"device={device}  band={fcut}  out_dir={out_dir}")

    base_path = out_dir / BASE_NAME
    if args.stage in ("base", "all"):
        print("[stage 1] pre-training base on S01-S07 ...")
        model, acc = pretrain_base(args.data_dir, device, fcut=fcut,
                                   max_epochs=args.max_epochs, seed=args.seed, verbose=args.verbose)
        torch.save(model.state_dict(), base_path)
        print(f"  base val_acc {acc:.2f}%  ->  {base_path}")

    if args.stage in ("finetune", "all"):
        state = torch.load(base_path, map_location=device)
        print(f"[stage 2] fine-tuning {'SE+Dense' if args.dense else 'SE only'} "
              f"for subjects {subjects[0]}-{subjects[-1]} ...")
        accs = []
        for s in subjects:
            model, acc, n = finetune_subject(state, s, args.data_dir, device, dense=args.dense,
                                             fcut=fcut, max_epochs=args.max_epochs,
                                             seed=args.seed, verbose=args.verbose)
            path = out_dir / _tuned_name(s, args.dense)
            torch.save(model.state_dict(), path)
            accs.append(acc)
            print(f"  S{s:02d}: val_acc {acc:5.2f}%  (trainable={n})  ->  {path.name}")
        if accs:
            print(f"  mean val_acc {sum(accs) / len(accs):.2f}%")


def _parse_subjects(spec: str):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


if __name__ == "__main__":
    main()
