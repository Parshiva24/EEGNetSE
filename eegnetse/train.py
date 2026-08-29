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
import csv
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from eegnetse.core import (
    BANDPASS, DATA_DIR, MODEL_DIR, ONLINE_SUBJECTS,
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
def _subject_loaders(subject, data_dir, fcut, val_ratio, seed):
    """Train/val loaders from a subject's calibration trials (72 trials)."""
    Xtr, Ytr, _, _ = online_session(data_dir, subject)
    Xt, Xv, Yt, Yv = train_test_split(
        Xtr, Ytr, train_size=1 - val_ratio, random_state=seed, shuffle=True, stratify=Ytr)
    train_loader = make_loader(Xt, Yt, batch_size=32, shuffle=True, fcut=fcut)
    val_loader = make_loader(Xv, Yv, batch_size=len(Yv), fcut=fcut)
    return train_loader, val_loader


def _evaluate_online(model, subject, data_dir, device, fcut):
    """Accuracy of an in-memory model on a subject's held-out online session."""
    _, _, X_online, Y_online = online_session(data_dir, subject)
    loader = make_loader(X_online, Y_online, batch_size=len(Y_online), fcut=fcut)
    return _evaluate(model, loader, device)


def finetune_subject(base_state, subject, data_dir=DATA_DIR, device=None, *,
                     dense=True, fcut=BANDPASS, lr=1e-3, max_epochs=1000,
                     val_ratio=0.2, seed=0, verbose=False):
    device = device or get_device()
    set_seed(seed)
    train_loader, val_loader = _subject_loaders(subject, data_dir, fcut, val_ratio, seed)

    model = build_model().to(device)
    model.load_state_dict(copy.deepcopy(base_state))
    groups = ["se", "dense"] if dense else ["se"]
    n_trainable = set_trainable(model, groups)
    model, acc = train_model(model, train_loader, val_loader, device,
                             lr=lr, max_epochs=max_epochs, verbose=verbose)
    return model, acc, n_trainable


def train_from_scratch(subject, data_dir=DATA_DIR, device=None, *, fcut=BANDPASS,
                       lr=1e-3, max_epochs=1000, val_ratio=0.2, seed=0, verbose=False):
    """Regime (a): train the full EEGNet-SE from random init on one subject's
    72 calibration trials only (no cross-subject pre-training)."""
    device = device or get_device()
    set_seed(seed)
    train_loader, val_loader = _subject_loaders(subject, data_dir, fcut, val_ratio, seed)
    model = build_model().to(device)                 # random init, all layers trainable
    model, acc = train_model(model, train_loader, val_loader, device,
                             lr=lr, max_epochs=max_epochs, verbose=verbose)
    return model, acc


# --------------------------------------------------------------------------- #
# Ablation: from-scratch vs. transfer learning  (reviewer response, Table III)
# --------------------------------------------------------------------------- #
def run_ablation(data_dir=DATA_DIR, model_dir=MODEL_DIR, out_dir=None, device=None, *,
                 fcut=BANDPASS, subjects=None, lr=1e-3, max_epochs=1000, seed=0,
                 save_models=True, csv_path=None, verbose=False):
    """Compare the three training regimes on each subject's online session.

        (a) from_scratch : full EEGNet-SE trained on the subject's 72 calib trials
        (b) base_only    : subject-independent base (S01-S07), no fine-tuning
        (c) two_stage    : proposed base + SE/Dense fine-tuning  (committed checkpoint)

    Regimes (b) and (c) reuse the committed checkpoints in ``model_dir``; (a) is
    trained here.  Writes a per-subject CSV (Table III) and returns the rows.
    """
    from eegnetse.infer import evaluate_checkpoint

    device = device or get_device()
    subjects = list(subjects) if subjects is not None else list(ONLINE_SUBJECTS)
    model_dir = Path(model_dir)
    if save_models and out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'subj':>4} | {'(a) scratch':>11} | {'(b) base':>9} | {'(c) two-stage':>13}")
    print("-" * 48)
    for s in subjects:
        model, _ = train_from_scratch(s, data_dir, device, fcut=fcut, lr=lr,
                                      max_epochs=max_epochs, seed=seed, verbose=verbose)
        a = _evaluate_online(model, s, data_dir, device, fcut)
        if save_models and out_dir is not None:
            torch.save(model.state_dict(), Path(out_dir) / f"scratch_sub{s:02d}.pth")
        b = evaluate_checkpoint(model_dir / BASE_NAME, s, data_dir, device, fcut)
        c = evaluate_checkpoint(model_dir / _tuned_name(s, True), s, data_dir, device, fcut)
        rows.append({"subject": s, "from_scratch": a, "base_only": b, "two_stage": c})
        print(f"S{s:02d}  | {a:11.2f} | {b:9.2f} | {c:13.2f}")

    def _col(key):
        return np.array([r[key] for r in rows])

    print("-" * 48)
    means = {k: _col(k).mean() for k in ("from_scratch", "base_only", "two_stage")}
    stds = {k: _col(k).std() for k in ("from_scratch", "base_only", "two_stage")}
    print(f"mean | {means['from_scratch']:11.2f} | {means['base_only']:9.2f} | {means['two_stage']:13.2f}")
    print(f"std  | {stds['from_scratch']:11.2f} | {stds['base_only']:9.2f} | {stds['two_stage']:13.2f}")

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject", "from_scratch", "base_only", "two_stage"])
            for r in rows:
                w.writerow([f"S{r['subject']:02d}", f"{r['from_scratch']:.2f}",
                            f"{r['base_only']:.2f}", f"{r['two_stage']:.2f}"])
            w.writerow(["mean", f"{means['from_scratch']:.2f}",
                        f"{means['base_only']:.2f}", f"{means['two_stage']:.2f}"])
            w.writerow(["std", f"{stds['from_scratch']:.2f}",
                        f"{stds['base_only']:.2f}", f"{stds['two_stage']:.2f}"])
        print(f"\nwrote {csv_path}")
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description="Train EEGNet-SE (base + per-subject fine-tuning).")
    p.add_argument("--stage", choices=["base", "finetune", "all", "ablation"], default="all")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--model-dir", default=str(MODEL_DIR),
                   help="committed checkpoints used for regimes (b)/(c) in the ablation")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "outputs"))
    p.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
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

    if args.stage == "ablation":
        print("[ablation] from-scratch (a) vs base-only (b) vs two-stage (c) ...")
        run_ablation(args.data_dir, args.model_dir, out_dir, device, fcut=fcut,
                     subjects=subjects, max_epochs=args.max_epochs, seed=args.seed,
                     csv_path=Path(args.results_dir) / "table3_ablation.csv",
                     verbose=args.verbose)
        return

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
