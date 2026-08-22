"""Stage 2: subject-specific fine-tuning + online evaluation over S08-S20.

Loads a pre-trained base model, fine-tunes it per subject on that subject's
calibration trials, evaluates on the online session, and writes a per-subject
accuracy table (CSV) plus the mean +/- SD.

Example
-------
    python scripts/finetune_eval.py --model eegnetse --finetune se_dense -v
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run without install

import torch

from eegnetse.cli import add_config_args, config_from_args
from eegnetse.config import set_seed, get_device, ONLINE_SUBJECTS
from eegnetse.training import finetune_subject, evaluate_online, FINETUNE_CONFIGS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--base-ckpt", default=None,
                        help="Base model checkpoint (default: outputs/base_<model>.pth).")
    parser.add_argument("--finetune", default="se_dense", choices=list(FINETUNE_CONFIGS),
                        help="Which layer groups to fine-tune.")
    args = parser.parse_args()

    config = config_from_args(args)
    set_seed(config.seed)
    device = get_device()

    ckpt = args.base_ckpt or (config.output_dir / f"base_{config.model}.pth")
    base_state = torch.load(ckpt, map_location=device)
    print(f"Device: {device} | base: {ckpt} | finetune: {args.finetune}")

    results = {}
    n_trainable = None
    for sub in ONLINE_SUBJECTS:
        set_seed(config.seed)  # identical init/split per subject, as in the notebooks
        model, n_trainable = finetune_subject(
            config, device, sub, base_state, finetune_config=args.finetune, verbose=args.verbose
        )
        acc = evaluate_online(model, config, device, sub)
        results[sub] = acc
        print(f"  S{sub:02d}: {acc:.2f}%")

    accs = list(results.values())
    print(f"\nMean {mean(accs):.2f} +/- {pstdev(accs):.2f}%  "
          f"(n={len(accs)} subjects, trainable params={n_trainable})")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    out = config.output_dir / f"finetune_{config.model}_{args.finetune}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "online_accuracy"])
        for sub, acc in results.items():
            w.writerow([f"S{sub:02d}", f"{acc:.4f}"])
        w.writerow(["mean", f"{mean(accs):.4f}"])
        w.writerow(["std", f"{pstdev(accs):.4f}"])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
