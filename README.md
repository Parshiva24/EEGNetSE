# EEGNet-SE — Motor-Imagery Direction Decoding

Subject-specific deep learning for **left/right motor-imagery (MI) direction
decoding** from EEG, built on EEGNet with squeeze-and-excitation *ranking*
layers over electrodes and feature maps. Reference implementation for the
manuscript *"Subject Specific Deep Learning Model for Motor Imagery Direction
Decoding"* (sensors-4477952).

> This repository was refactored from a set of exploratory notebooks into an
> installable package. The originals are archived under [`legacy/`](legacy/);
> see [`CHANGES.md`](CHANGES.md) for what changed.

## The method in one paragraph

A subject-independent **base model** is pre-trained on the pooled calibration
trials of subjects S01–S07. For each target subject (S08–S20) the base model is
then **fine-tuned** on that subject's 72 calibration trials — only a chosen set
of layers is updated, the rest frozen — and evaluated on that subject's held-out
online (closed-loop feedback) session. No online trial is ever used for training
or model selection.

## Install

```bash
# 1. install PyTorch for your CUDA version from https://pytorch.org, then:
pip install -e .
```

Point the package at your `.mat` data (files `S01_mitrials.mat` … `S20_mitrials.mat`):

```bash
export EEGNETSE_DATA=/path/to/data     # or pass --data-dir to any script
```

By default the data is expected in `./data` next to this README.

## Package layout

```
eegnetse/
├── config.py            # Config dataclass, seeding, device, dataset constants
├── data/
│   ├── io.py            # load .mat, calibration/online/LOSO dataset builders
│   └── preprocessing.py # baseline correction, notch+band-pass, multiband, EA
├── models/
│   ├── layers.py        # SE ranking blocks, constrained conv/linear
│   ├── eegnet.py        # EEGNet + proposed EEGNet-SE (one flag-driven class)
│   ├── fbcnet.py        # FBCNet baseline
│   └── mvca.py          # MVCA baseline (stub — experiment A4)
└── training/
    ├── tensors.py       # numpy trials -> DataLoader
    ├── trainer.py       # train / evaluate loops
    ├── earlystop.py     # unified early stopping
    └── protocol.py      # two-stage pretrain + fine-tune + online eval
scripts/                 # thin runnable entry points (replace the notebooks)
legacy/                  # archived original notebooks
```

## Usage

```bash
# Stage 1 — pre-train the subject-independent base model on S01–S07
python scripts/pretrain_base.py --model eegnetse -v

# Baseline — evaluate the unadapted base model on the online sessions
python scripts/evaluate_base.py --model eegnetse

# Stage 2 — subject-specific fine-tuning + online evaluation (writes a CSV)
python scripts/finetune_eval.py --model eegnetse --finetune se_dense -v
```

Swap `--model eegnet` for the no-ranking baseline or `--model fbcnet` for FBCNet.
Fine-tuning configurations (`--finetune`) are defined in
`eegnetse.training.FINETUNE_CONFIGS`:

| name | layers updated on target subject |
|---|---|
| `dense` | classifier only |
| `se_dense` | electrode + filter ranking, classifier |
| `se_conv_dense` | ranking, separable conv, classifier |
| `electrode_se_dense` | electrode ranking, classifier |
| `continued` | all parameters (full fine-tuning) |

Models without ranking layers (`eegnet`, `fbcnet`) only have the shared layers,
so run them with `--finetune continued` (or `dense`).

### As a library

```python
from eegnetse import Config, set_seed, get_device
from eegnetse.training import pretrain_base, finetune_subject, evaluate_online

cfg = Config(model="eegnetse", data_dir="/path/to/data")
set_seed(cfg.seed); dev = get_device()

base = pretrain_base(cfg, dev)
model, n_ft = finetune_subject(cfg, dev, subject=14, base_state=base.state_dict(),
                               finetune_config="se_dense")
print(evaluate_online(model, cfg, dev, subject=14))
```

## Reproducibility

Every hyper-parameter lives in `eegnetse.config.Config`; a run is fully described
by one instance. `set_seed` seeds Python, NumPy and PyTorch.

## Paper revision status

The manuscript received a major revision. Outstanding experiments (train-from-
scratch, batch-norm ablation, leave-one-subject-out CV, MVCA baseline, complexity
table, EEG-example figure) are tracked in
`../Revision-Checklist.md`. The refactored protocol is model- and
configuration-agnostic so those experiments slot in as thin scripts on top of it.
