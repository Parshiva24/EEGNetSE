"""Stage 1: pre-train the subject-independent base model on S01-S07 calibration.

Example
-------
    python scripts/pretrain_base.py --model eegnetse -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run without install

import torch

from eegnetse.cli import add_config_args, config_from_args
from eegnetse.config import set_seed, get_device
from eegnetse.training import pretrain_base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args()

    config = config_from_args(args)
    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device} | model: {config.model}")

    model = pretrain_base(config, device, verbose=args.verbose)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = config.output_dir / f"base_{config.model}.pth"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved base model -> {ckpt}")


if __name__ == "__main__":
    main()
