"""Shared argparse helpers so every script exposes the same config overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

from eegnetse.config import Config


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory of S*_mitrials.mat files (default: $EEGNETSE_DATA or ./data).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="eegnetse",
                        choices=["eegnet", "eegnetse", "fbcnet", "mvca"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3, dest="learning_rate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--se-reduction", type=int, default=3)
    parser.add_argument("-v", "--verbose", action="store_true")


def config_from_args(args) -> Config:
    kwargs = dict(
        output_dir=args.output_dir,
        model=args.model,
        seed=args.seed,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        se_reduction=args.se_reduction,
    )
    if args.data_dir is not None:
        kwargs["data_dir"] = args.data_dir
    return Config(**kwargs)
