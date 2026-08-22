"""Evaluate the *unadapted* base model on every online session (S08-S20).

This is the subject-independent baseline row -- no subject-specific fine-tuning.

Example
-------
    python scripts/evaluate_base.py --model eegnetse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run without install

import torch

from eegnetse.cli import add_config_args, config_from_args
from eegnetse.config import set_seed, get_device, ONLINE_SUBJECTS
from eegnetse.models import build_model
from eegnetse.training import evaluate_online


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--base-ckpt", default=None,
                        help="Base checkpoint (default: outputs/base_<model>.pth).")
    args = parser.parse_args()

    config = config_from_args(args)
    set_seed(config.seed)
    device = get_device()

    ckpt = args.base_ckpt or (config.output_dir / f"base_{config.model}.pth")
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    accs = []
    for sub in ONLINE_SUBJECTS:
        acc = evaluate_online(model, config, device, sub)
        accs.append(acc)
        print(f"  S{sub:02d}: {acc:.2f}%")
    print(f"\nUnadapted base mean {mean(accs):.2f} +/- {pstdev(accs):.2f}%")


if __name__ == "__main__":
    main()
